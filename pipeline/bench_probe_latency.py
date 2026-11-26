"""Measure batch-one CPU logit latency and resources with real hidden vectors already in RAM.

Defaults: saved Qwen 0.8B C/13 source probe and HotpotQA technical-pilot states.
The target-fitted pilot readout has the same shape and kernel; no fitting is performed here.
Latency is in microseconds per call. Inputs keep the saved dtype; normalization with
float64 weights is inside the timer. RSS includes the interpreter, NumPy and the harness.
"""

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
if __name__ == '__main__':
    for var in THREAD_VARS:
        os.environ[var] = '1'  # Only in this benchmark process, before importing NumPy.

import numpy as np

from apply_probe_model import load_probe_model
from hotpotqa import read_json
from latency_common import COUNT, HERE, PILOT, SEED, digest, distribution, new_report_path, workload, write_report
from probe_refit import score


def process_memory():
    """Current and lifetime-peak resident bytes of this process, not the probe alone."""
    rss = peak = None
    if sys.platform == 'win32':
        import ctypes as c
        from ctypes import wintypes as wt

        class Counters(c.Structure):
            _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [(k, c.c_size_t) for k in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

        handle = c.windll.kernel32.GetCurrentProcess
        handle.restype = wt.HANDLE
        get_mem = c.windll.psapi.GetProcessMemoryInfo
        get_mem.argtypes = (wt.HANDLE, c.POINTER(Counters), wt.DWORD)
        get_mem.restype = wt.BOOL
        counters = Counters()
        counters.cb = c.sizeof(counters)
        if get_mem(handle(), c.byref(counters), counters.cb):
            rss, peak = counters.WorkingSetSize, counters.PeakWorkingSetSize
    elif sys.platform.startswith('linux'):
        status = Path('/proc/self/status')
        if status.is_file():
            fields = {line.split(':')[0]: int(line.split()[1]) * 1024
                      for line in status.read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}
            rss, peak = fields.get('VmRSS'), fields.get('VmHWM')
    elif sys.platform == 'darwin':
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    return {'rss_bytes': rss, 'lifetime_peak_rss_bytes': peak}


def load_vectors(run, probe, items):
    point = probe['info']['point']
    layer_id = int(probe['info']['layer_id'] if 'layer_id' in probe['info'] else probe['info']['layer'])
    layers = read_json(run / 'config.json')['layers']
    h = np.load(run / f'hidden_{point.lower()}.npy', mmap_mode='r', allow_pickle=False)
    if h.ndim != 3 or h.shape[1] != len(layers) or layers.count(layer_id) != 1 or h.shape[2] != len(probe['mu']):
        raise ValueError('selected point/layer or hidden dimension does not match the saved probe')
    indices = [item['row'] for item in items]
    if min(indices) < 0 or max(indices) >= h.shape[0]:
        raise ValueError('workload row outside hidden array')
    # Advanced indexing copies only the selected vectors; no memory mapping in the timed loop.
    bank = np.ascontiguousarray(h[indices, layers.index(layer_id), :])
    if not np.isfinite(bank).all():
        raise ValueError('non-finite vector in timing workload')
    return bank, layer_id


def cpu_cost(w, mu, sd, views):
    # Separate pass: do not charge per-call clocks, quantile computation or RSS queries to CPU time.
    it = cycle(views)
    count = 0
    start = time.perf_counter_ns()
    cpu0 = time.process_time_ns()
    while time.perf_counter_ns() - start < 1_000_000_000:
        for _ in range(1024):
            score(w, next(it), mu, sd)
        count += 1024
    cpu = time.process_time_ns() - cpu0
    wall = time.perf_counter_ns() - start
    return {'calls': count, 'process_cpu_us_per_call_including_loop': cpu / (count * 1000),
            'wall_seconds': wall / 1e9, 'average_busy_cores': cpu / wall}


def traced_call_peak(w, h, mu, sd):
    # Separate from latency and CPU timing. Only allocations visible to tracemalloc,
    # including NumPy allocations it tracks; this is not an OS-level peak or total RSS.
    import tracemalloc
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        score(w, h, mu, sd)
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=HERE / 'out/main_qwen08_q4/probe_model.npz')
    ap.add_argument('--run-dir', type=Path, default=PILOT, help='run with meta.jsonl, config.json and hidden states')
    ap.add_argument('--count', type=int, default=COUNT, help='distinct measured vectors; same sampling as Jev')
    ap.add_argument('--repeats', type=int, default=1000, help='timed individual calls per distinct vector')
    ap.add_argument('--out', type=Path, help='new report file; defaults to out/latency/<timestamped name>')
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error('--repeats must be positive')
    path = new_report_path('probe', args.out)
    started = datetime.now(timezone.utc).isoformat()
    warm, measured = workload(args.run_dir, args.count)
    model_sha = digest(args.model.read_bytes())
    memory = {'before_weights': process_memory()}
    probe = load_probe_model(args.model)
    w, mu, sd = probe['w'], probe['mu'], probe['sd']
    memory['after_weights'] = process_memory()
    bank, layer_id = load_vectors(args.run_dir, probe, warm + measured)
    warm_views = tuple(bank[:len(warm)])
    views = tuple(bank[len(warm):])
    for _ in range(100):
        for h in warm_views:
            score(w, h, mu, sd)
    if not all(np.isfinite(score(w, h, mu, sd)) for h in views):
        raise ValueError('non-finite score')
    memory['after_vectors_and_warmup'] = process_memory()

    now = time.perf_counter_ns
    clock_pairs = []
    for _ in range(1000):
        t0 = now()
        clock_pairs.append(now() - t0)
    elapsed = np.empty((args.repeats, len(views)), dtype=np.int64)
    for durations in elapsed:
        for i, h in enumerate(views):
            t0 = now()
            score(w, h, mu, sd)
            dt = now() - t0
            durations[i] = dt
    costs = cpu_cost(w, mu, sd, views)
    memory['after_measurement'] = process_memory()
    temporary_bytes = traced_call_peak(w, views[0], mu, sd)
    report = {'kind': 'probe_single_cpu_logit', 'started_utc': started,
              'kernel': 'b + ((h - mu) / sd) @ weights; resident input, batch=1, no sigmoid',
              'model': str(args.model), 'model_sha256': model_sha, 'run_dir': str(args.run_dir),
              'point': probe['info']['point'], 'model_layer': layer_id, 'D': int(mu.size),
              'input_dtype': str(bank.dtype), 'weight_dtype': str(w.dtype), 'sampling_seed': SEED,
              'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'requested_thread_settings': {k: os.environ.get(k) for k in THREAD_VARS},
              'warmup_calls': 100 * len(warm_views), 'repeats_per_vector': args.repeats,
              'latency_us': distribution(elapsed.ravel() / 1000), 'back_to_back_timer_ns': distribution(clock_pairs),
              'cpu_separate_pass': costs, 'process_memory_snapshots': memory,
              'one_call_incremental_tracemalloc_peak_bytes': temporary_bytes,
              'model_file_bytes': args.model.stat().st_size, 'learned_coefficients': int(w.size),
              'learned_weights_bytes': w.nbytes, 'normalization_bytes': mu.nbytes + sd.nbytes,
              'model_arrays_bytes': w.nbytes + mu.nbytes + sd.nbytes,
              'one_input_vector_bytes': bank[0].nbytes, 'benchmark_vector_bank_bytes': bank.nbytes,
              'per_input': [{**row, 'latency_us': distribution(elapsed[:, i] / 1000)}
                            for i, row in enumerate(measured)]}
    write_report(path, report)
    print(f'Logit latency (us): {report["latency_us"]}; model arrays: {report["model_arrays_bytes"]} bytes; {path}')


if __name__ == '__main__':
    main()
