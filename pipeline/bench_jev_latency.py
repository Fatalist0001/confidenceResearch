"""Measure client-observed Jev latency, one answer/three Nouls per request.

Run after other users of the gateway quota are idle. Requires curl >= 7.84.
Uses the disjoint HotpotQA technical pilot, not the quality pilot's 199 test items.
Three warmups + 30 measured requests by default. Each invocation sends live requests.
After an interruption, --resume FILE starts a new warmed session for unresolved IDs only.
Latency is curl's time_total (ms), excluding client pacing, including the network and gateway.
New connections and server-advertised Server-Timing (if present) are recorded separately.
"""

import argparse
import json
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hotpotqa import read_jsonl
from jev_hotpot_pilot import ENDPOINT, MODEL, gateway_key, jev_score, wire_body
from latency_common import COUNT, HERE, PILOT, SEED, WARMUP, digest, distribution, new_report_path, workload, write_report


def make_jobs(count):
    warm, measured = workload(PILOT, count)
    prompts = read_jsonl(HERE / 'data/hotpotqa_qwen08_v2/pilot.jsonl')
    meta = read_jsonl(PILOT / 'meta.jsonl')
    jobs = []
    for i, item in enumerate(warm + measured):
        n = item['row']
        if not (0 <= n < len(prompts) and 0 <= n < len(meta) and
                prompts[n]['id'] == meta[n]['id'] == item['id'] and meta[n]['row'] == n):
            raise ValueError('HotpotQA input / answer ID mismatch')
        body = wire_body(prompts[n]['question'], meta[n]['answer'])
        jobs.append(({**item, 'phase': 'warmup' if i < len(warm) else 'measure',
                      'request_bytes': len(body), 'request_sha256': digest(body)}, body))
    return jobs


def curl_config(jobs, key, tmp):
    # A single process keeps libcurl's connection pool across --next. The bearer and
    # request bodies travel to curl on stdin, never in argv or in the report.
    if not key.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError('invalid characters in gateway key')
    lines = []
    for i, (_, body) in enumerate(jobs):
        if i:
            lines.append('next')
        lines.extend(['silent', 'show-error', 'fail-with-body'])
        options = [('proto', '=https'), ('connect-timeout', '12'), ('max-time', '45'),
                   ('header', 'Authorization: Bearer ' + key), ('header', 'Content-Type: application/json'),
                   ('url', ENDPOINT), ('data-binary', body.decode('utf-8')),
                   ('output', (tmp / f'{i}.body').as_posix()), ('dump-header', (tmp / f'{i}.hdr').as_posix()),
                   ('write-out', r'%{json}\n')]
        lines.extend(f'{k} = {json.dumps(v, ensure_ascii=False)}' for k, v in options)
    return ('\n'.join(lines) + '\n').encode('utf-8')


def diagnostic_headers(path):
    out = {}
    if path.exists():
        for line in path.read_bytes().decode('latin-1').splitlines():
            if line.startswith('HTTP/'):
                out.clear()  # Ignore proxy CONNECT and interim 1xx headers.
            name, _, value = line.partition(':')
            if name.lower() in ('server-timing', 'retry-after', 'cf-cache-status', 'age'):
                out[name.lower()] = value.strip()[:512]
    return out


def record(item, timing, tmp, i):
    status = int(timing['http_code'])
    r = {**item, 'status': status, 'curl_exit': timing['exitcode'], 'http_version': timing['http_version'],
         'new_connections': timing['num_connects'], 'ok': False,
         'response_headers': diagnostic_headers(tmp / f'{i}.hdr')}
    for label, field in [('total_ms', 'time_total'), ('first_byte_ms', 'time_starttransfer'),
                         ('dns_end_ms', 'time_namelookup'), ('connect_end_ms', 'time_connect'),
                         ('tls_end_ms', 'time_appconnect')]:
        r[label] = 1000 * float(timing[field])
    if status != 200 or r['curl_exit'] != 0:
        r['error'] = 'http_or_transport_error'
        return r
    try:
        data = json.loads((tmp / f'{i}.body').read_bytes())
        rec = {'model': data['model'], 'nouls': {k: v['noul'] for k, v in data['answers'].items()},
               'input_tokens': (data.get('usage') or {}).get('input_tokens')}
        jev_score(rec)  # Wrong model or invalid probabilities do not count as a successful verification.
        r.update(ok=True, reported_model=rec['model'], input_tokens=rec['input_tokens'])
    except (KeyError, TypeError, ValueError, AttributeError, OSError):
        r['error'] = 'invalid_jev_response'
    return r


def resume_plan(path, planned, version):
    if path is None:
        return [], [], planned, None
    raw = path.read_bytes()
    prior = json.loads(raw)
    contract = {'kind': 'jev_client_latency', 'endpoint': ENDPOINT, 'model': MODEL, 'concurrency': 1,
                'max_requests_per_minute': 4, 'configured_retries': 0, 'sampling_seed': SEED,
                'dataset': PILOT.name, 'planned_measured': len(planned) - WARMUP, 'curl': version,
                'platform': platform.platform()}
    if any(prior.get(k) != v for k, v in contract.items()):
        raise ValueError('previous report used a different workload or transport')
    by_id = {item['id']: item for item, _ in planned}
    rows = [{'session': 1, **r} for r in prior['attempts']]
    done = set()
    for r in rows:
        expected = by_id.get(r['id'])
        if expected is None or any(r[k] != v for k, v in expected.items()):
            raise ValueError('previous row differs from the current request')
        if r['ok'] and r['phase'] == 'measure':
            if r['id'] in done:
                raise ValueError('duplicate successful measured ID in previous report')
            done.add(r['id'])
    if len(done) != prior['valid_measured'] or len(done) == contract['planned_measured']:
        raise ValueError('inconsistent or already complete report; nothing to resume')
    sessions = prior.get('sessions', [{'started_utc': prior['started_utc'],
                                        'curl_process_exit': prior['curl_process_exit'], 'attempts': len(rows)}])
    return (rows, sessions, [j for j in planned if j[0]['id'] not in done],
            {'file': path.name, 'sha256': digest(raw)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count', type=int, default=COUNT, help='distinct measured answers, excluding 3 warmups')
    ap.add_argument('--out', type=Path, help='new JSON file; defaults to out/latency/<timestamped name>')
    ap.add_argument('--resume', type=Path, help='previous incomplete report for the same --count')
    args = ap.parse_args()
    planned = make_jobs(args.count)
    path = new_report_path('jev', args.out)
    version = subprocess.check_output(['curl', '-q', '--version'], text=True).splitlines()[0]
    match = re.match(r'curl (\d+)\.(\d+)', version)
    if not match or tuple(map(int, match.groups())) < (7, 84):
        raise SystemExit('curl >= 7.84 required for paced sequential transfers')
    previous, sessions, jobs, parent = resume_plan(args.resume, planned, version)
    key = gateway_key()
    started = datetime.now(timezone.utc).isoformat()
    print(f'This session: {WARMUP} warmup + {len(jobs)-WARMUP} measured; sequential, 4 per minute.', flush=True)
    with tempfile.TemporaryDirectory(prefix='.jev-', dir=path.resolve().parent) as name:
        tmp = Path(name)
        proc = subprocess.run(['curl', '-q', '--fail-early', '--rate', '4/m', '--config', '-'],
                              input=curl_config(jobs, key, tmp), capture_output=True)
        timings = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        if len(timings) > len(jobs):
            raise ValueError('curl returned more transfers than requested')
        current = [dict(record(jobs[i][0], timing, tmp, i), session=len(sessions) + 1)
                   for i, timing in enumerate(timings)]
    rows = previous + current
    sessions.append({'started_utc': started, 'curl_process_exit': proc.returncode, 'attempts': len(current)})
    good = [r for r in rows if r['phase'] == 'measure' and r['ok']]
    reused = [r for r in good if r['new_connections'] == 0]
    first = {}
    for r in rows:
        if r['phase'] == 'measure':
            first.setdefault(r['id'], r)
    attempted = {r['id'] for r in rows}
    complete = len(good) == args.count and all(r['ok'] for r in current) and proc.returncode == 0
    report = {'kind': 'jev_client_latency', 'started_utc': sessions[0]['started_utc'], 'endpoint': ENDPOINT, 'model': MODEL,
              'curl': version, 'platform': platform.platform(), 'concurrency': 1, 'max_requests_per_minute': 4,
              'configured_retries': 0, 'sampling_seed': SEED, 'dataset': PILOT.name,
              'warmups': sum(r['phase'] == 'warmup' for r in rows), 'warmups_per_session': WARMUP,
              'sessions': sessions, 'resumed_from': parent,
              'planned_measured': args.count, 'valid_measured': len(good), 'complete': complete,
              'failed_attempts': sum(not r['ok'] for r in rows),
              'curl_process_exit': proc.returncode, 'not_attempted_ids': [j[0]['id'] for j in planned if j[0]['id'] not in attempted],
              'first_attempt_success_total_ms': distribution([r['total_ms'] for r in first.values() if r['ok']]),
              'post_warmup_total_ms': distribution([r['total_ms'] for r in good]),
              'reused_connection_total_ms': distribution([r['total_ms'] for r in reused]), 'attempts': rows}
    write_report(path, report)
    print(f'Valid measured: {len(good)}/{args.count}; total ms: {report["post_warmup_total_ms"]}; {path}')
    if not complete:
        raise SystemExit('Incomplete run; see the report for HTTP/transport errors and omitted requests.')


if __name__ == '__main__':
    main()
