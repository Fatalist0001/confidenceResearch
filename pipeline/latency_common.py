"""Shared workload and reporting for the two standalone latency measurements."""

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PILOT = HERE / 'out/pilot_qwen08_q4_hotpotqa_v2'
COUNT = 30
WARMUP = 3
SEED = 42


def workload(run, count=COUNT):
    """Disjoint warmup; one measured item per equal-size prompt-length stratum."""
    rows = []
    with (run / 'meta.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if (r['status'] == 'ok' and r['generated'] and not r['truncated'] and r['answer'].strip()):
                rows.append({'id': r['id'], 'row': int(r['row']), 'qwen_prompt_tokens': int(r['pos_a']) + 1})
    if not 1 <= count <= len(rows) - WARMUP:
        raise ValueError(f'count must be in [1, {len(rows) - WARMUP}]')
    if len({r['id'] for r in rows}) != len(rows) or len({r['row'] for r in rows}) != len(rows):
        raise ValueError('duplicate ID or row in eligible workload')
    rng = random.Random(SEED)
    rows.sort(key=lambda r: r['id'])
    rng.shuffle(rows)
    warm = rows[:WARMUP]
    pool = sorted(rows[WARMUP:], key=lambda r: (r['qwen_prompt_tokens'], r['id']))
    n = len(pool)
    measured = [pool[rng.randrange(i*n//count, (i+1)*n//count)] for i in range(count)]
    rng.shuffle(measured)
    return warm, measured


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {'n': 0, 'p50': None, 'p95': None, 'mean': None}
    p50, p95 = np.percentile(a, [50, 95])
    return {'n': int(a.size), 'p50': float(p50), 'p95': float(p95), 'mean': float(a.mean())}


def new_report_path(kind, override):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(override) if override else HERE / 'out/latency' / f'{kind}_{stamp}.json'
    if path.exists():
        raise FileExistsError(f'{path} already exists; choose a new report name')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_report(path, report):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()
