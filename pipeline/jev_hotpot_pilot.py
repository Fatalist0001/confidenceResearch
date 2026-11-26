"""Paired Jev pilot on saved Qwen answers to HotpotQA; no generator is needed.

Default: prepare the pilot and show the local side without contacting the API.
`smoke`: score the first test question. `run`: score the remaining test questions.
The gateway credentials are read from the project's .env (JEV_*).
"""

import argparse
import csv
import getpass
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from baseline import auroc, ci
from hotpotqa import INSTRUCTION, PROMPT_PREFIX, read_json, read_jsonl, require, sha256, write_json
from probe_refit import fit_l2, pick_lambda, score, stratified_split


HERE = Path(__file__).resolve().parent
RUN = HERE / 'out/main_qwen08_q4_hotpotqa_v2'
INPUT = HERE / 'data/hotpotqa_qwen08_v2/main.jsonl'
OUT = HERE / 'out/jev_hotpot_pilot_v1'
BASE_URL = 'https://100milionseek.top/v1'
MODEL = 'jev-1.13-free'  # ID returned by this gateway's authenticated GET /v1/models
ENDPOINT = BASE_URL + '/systemone'
COOKBOOK = 'https://docs.typesafe.ai/cookbooks/verify_llm_outputs'


def noul(instructions, yes, no):
    return {'type': 'noul', 'instructions': instructions, 'criteria': {'true': yes, 'false': no}}


# The three questions and the decline branch are from the official RAG cookbook.
QUESTIONS = {
    'answerable': noul('Do the paragraphs state every fact needed to answer the question?',
                       'each step from the question to its answer is stated in the paragraphs',
                       'at least one step from the question to its answer is not stated in the paragraphs'),
    'declines': noul('Does the candidate response decline to answer, or say the information is missing?',
                     'the candidate says it cannot answer or that the paragraphs do not say',
                     'the candidate gives an answer'),
    'supported': noul("Following the paragraphs step by step from the question, do you arrive at the candidate's answer?",
                      "the paragraphs connect the question to the candidate's answer through stated facts",
                      "the paragraphs do not connect the question to the candidate's answer"),
}


def wire_body(prompt, candidate):
    """Allowlist for the API: original question, complete evidence, instruction and candidate only."""
    suffix = '\n' + INSTRUCTION
    require(prompt.startswith(PROMPT_PREFIX) and prompt.endswith(suffix), 'unexpected generator prompt')
    paragraphs, sep, question = prompt[len(PROMPT_PREFIX):-len(suffix)].rpartition('\n\nQuestion: ')
    require(sep and paragraphs.startswith('[1] ') and question.strip() and candidate.strip(), 'empty request field')
    state = {'request': question, 'paragraphs': paragraphs, 'candidate_response': candidate,
             'response_requirement': 'Use the supplied paragraphs to answer the question. ' + INSTRUCTION}
    return json.dumps({'model': MODEL, 'state': state, 'questions': QUESTIONS},
                      ensure_ascii=False, sort_keys=True).encode('utf-8')


def prepare():
    scores_file = RUN / 'transfer_scores.csv'
    hidden = RUN / 'hidden_c.npy'
    prior = read_json(RUN / 'refit_curve.json')
    hashes = {p.name: sha256(p) for p in (scores_file, hidden, INPUT, RUN / 'meta.jsonl')}
    for p, key in ((scores_file, 'scores_sha256'), (hidden, 'hidden_sha256')):
        require(hashes[p.name] == prior['provenance'][key], f'input differs from refit study: {p}')
    with scores_file.open(encoding='utf-8', newline='') as f:
        cohort = [r for r in csv.DictReader(f) if r['analysis_included'] == '1']
    original = np.array([int(r['row']) for r in cohort])
    y = np.array([int(r['hotpot_em']) for r in cohort])
    ids = [r['id'] for r in cohort]
    meta, prompts = read_jsonl(RUN / 'meta.jsonl'), read_jsonl(INPUT)
    require(len(set(ids)) == len(y) == 499 and y.sum() == 161, 'unexpected primary cohort')
    for i, qid in zip(original, ids):
        require(meta[i]['row'] == i and prompts[i]['id'] == meta[i]['id'] == qid, 'row/ID mismatch')

    # Exactly the existing scratch analysis at n_train=300, repeat zero, not the best repeat.
    X = np.load(hidden, mmap_mode='r')[original, 0, :].astype(np.float64)
    rng = np.random.default_rng(1000)
    tr, te = stratified_split(y, 300, rng)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    Xtr = (X[tr] - mu) / sd
    lam = pick_lambda(Xtr, y[tr], rng, prior['lambdas'], prior['val_frac'])
    w = fit_l2(Xtr, y[tr].astype(np.float64), lam)
    logits = score(w, X[te], mu, sd)
    rows = []
    for j, logit in zip(te, logits):
        i = int(original[j])
        body = wire_body(prompts[i]['question'], meta[i]['answer'])
        rows.append({'id': ids[j], 'row': i, 'em': int(y[j]), 'body': body,
                     'request_sha256': hashlib.sha256(body).hexdigest(), 'adapted': float(logit),
                     'frozen': float(cohort[j]['frozen_probe_logit']), 'qwen_tokens': meta[i]['pos_c'] + 1})
    protocol = {
        'study': 'jev_hotpot_pilot_v1', 'endpoint': ENDPOINT, 'model': MODEL, 'recipe': COOKBOOK, 'questions': QUESTIONS,
        'jev_accept_score': '1 - answerable if declines >= 0.5 else supported',
        'primary': 'risk = fraction with hotpot_em=0 among top ceil(n/2); tie by original row ascending',
        'secondary': 'AUROC for hotpot_em=1; higher score means more likely correct',
        'comparison': 'adapted vs zero-shot Jev; frozen source probe is a descriptive control',
        'intervals': 'paired question bootstrap, percentile 95%, 1000 draws, seed 42; conditional on fit',
        'inputs_sha256': hashes, 'split': 'probe_refit.py: n_train=300, repeat=0, rng seed=1000',
        'fit': {'point': 'C', 'model_layer': 13, 'lambda': lam, 'val_fraction': prior['val_frac'],
                'lambda_grid': prior['lambdas'], 'train_ids': [ids[j] for j in tr]},
        'test': [{k: r[k] for k in ('row', 'id', 'request_sha256')} for r in rows],
    }
    return rows, protocol


def measure(y, scores, original):
    k = (len(y) + 1) // 2
    keep = np.lexsort((original, -scores))[:k]
    wrong = int((y[keep] == 0).sum())
    return {'accepted': k, 'wrong': wrong, 'risk': wrong / k, 'auroc': auroc(y, scores)}


def paired_deltas(y, adapted, jev, original):
    def delta(idx):
        p = measure(y[idx], adapted[idx], original[idx])
        j = measure(y[idx], jev[idx], original[idx])
        return j['risk'] - p['risk'], p['auroc'] - j['auroc']

    rng = np.random.default_rng(42)
    d = np.asarray([delta(rng.integers(len(y), size=len(y))) for _ in range(1000)])
    risk, auc = delta(np.arange(len(y)))
    return {'risk_jev_minus_adapted': risk, 'risk_delta_ci95': ci(d[:, 0]),
            'auroc_adapted_minus_jev': auc, 'auroc_delta_ci95': ci(d[:, 1])}


def jev_score(rec):
    p = rec['nouls']
    require(rec['model'] in (MODEL, 'jev-1.13.0') and set(p) == set(QUESTIONS),
            f'unexpected response model ({rec["model"]!r}) or questions')
    require(all(type(v) in (int, float) and 0 <= v <= 1 for v in p.values()), 'invalid Noul probability')
    tokens = rec['input_tokens']
    require(tokens is None or (type(tokens) is int and tokens > 0), 'invalid Jev input token usage')
    return 1 - p['answerable'] if p['declines'] >= 0.5 else p['supported']


def ask(body, key):
    # Use the documented curl client. Both key and body go on stdin, never in argv or logs.
    config = ('header = ' + json.dumps('Authorization: Bearer ' + key) + '\n' +
              'data-binary = ' + json.dumps(body.decode('utf-8'), ensure_ascii=False) + '\n').encode('utf-8')
    for attempt in range(3):
        proc = subprocess.run(['curl', '-q', '-sS', '--proto', '=https', '--connect-timeout', '12',
                               '--max-time', '45', '--header', 'Content-Type: application/json', '--config', '-',
                               '--write-out', '\\n%{http_code}|%header{retry-after}|%header{x-quota-remaining}',
                               ENDPOINT], input=config, capture_output=True, check=True)
        raw, _, footer = proc.stdout.rpartition(b'\n')
        status, after, quota = footer.decode('ascii').strip().split('|')
        if status == '200':
            data = json.loads(raw)
            break
        delay = int(after) if after.isdecimal() else 60
        if status not in ('429', '503') or attempt == 2 or quota == '0' or delay > 90:
            raise SystemExit(f'Jev HTTP {status}; Retry-After={after or "n/a"}. Saved responses retained.')
        time.sleep(max(1, delay) + 1)
    rec = {'model': data['model'], 'nouls': {k: v['noul'] for k, v in data['answers'].items()},
           'input_tokens': (data.get('usage') or {}).get('input_tokens')}
    jev_score(rec)
    return rec


def gateway_key():
    names = ('JEV_BASE_URL', 'JEV_MODEL', 'JEV_API_KEY')
    env = HERE.parent / '.env'
    cfg = {}
    if env.is_file():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.partition('=')
            if sep and name.strip() in names:
                cfg[name.strip()] = value.strip().strip('\"\'')
    cfg.update({k: os.environ[k] for k in names if os.environ.get(k)})
    require(cfg.get('JEV_BASE_URL', BASE_URL).rstrip('/') == BASE_URL and cfg.get('JEV_MODEL', MODEL) == MODEL,
            'gateway/model differs from this pilot protocol')
    key = cfg.get('JEV_API_KEY')
    if not key and sys.stdin.isatty():
        key = getpass.getpass('Gateway API key (hidden): ')
    require(key and key.strip(), 'set JEV_API_KEY in .env or run in an interactive terminal')
    return key.strip()


def save_once(path, data):
    if path.exists():
        require(read_json(path) == data, f'{path.name} differs from saved pilot; refusing to overwrite')
    else:
        write_json(path, data)


def collect(rows, mode, directory):
    path = directory / 'jev_scores.jsonl'
    expected = {r['id']: r['request_sha256'] for r in rows}
    cache = {}
    for r in read_jsonl(path) if path.exists() else []:
        qid = r['id']
        require(qid not in cache and expected.get(qid) == r['request_sha256'], 'duplicate/stale cache row')
        jev_score(r)
        cache[qid] = r
    wanted = {'offline': [], 'smoke': rows[:1], 'run': rows}[mode]
    missing = [r for r in wanted if r['id'] not in cache]
    if missing:
        key = gateway_key()
        last_start = None
        with path.open('a', encoding='utf-8', newline='\n') as f:
            for row in missing:
                # Published free-tier limit is five requests per minute.
                if last_start is not None:
                    time.sleep(max(0, 12.2 - (time.monotonic() - last_start)))
                last_start = time.monotonic()
                rec = {'id': row['id'], 'request_sha256': row['request_sha256'], **ask(row['body'], key)}
                f.write(json.dumps(rec, allow_nan=False) + '\n')
                f.flush()
                cache[row['id']] = rec
                if len(cache) == 1 or len(cache) % 25 == 0:
                    print(f'Jev: {len(cache)}/{len(rows)}', flush=True)
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('offline', 'smoke', 'run'), nargs='?', default='offline')
    args = ap.parse_args()
    rows, protocol = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    save_once(OUT / 'protocol.json', protocol)  # freeze IDs, requests and metrics before contacting Jev
    y, original = (np.array([r[k] for r in rows]) for k in ('em', 'row'))
    scores = {k: np.array([r[k] for r in rows]) for k in ('adapted', 'frozen')}
    local = {k: measure(y, v, original) for k, v in scores.items()}
    print(f'Qwen3.5-0.8B | train=300, test={len(y)}, EM correct={y.sum()}, lambda={protocol["fit"]["lambda"]:g}')
    for k, v in local.items():
        print(f'{k}: {v["wrong"]}/{v["accepted"]} wrong at half coverage; AUROC={v["auroc"]:.4f}')
    # Order of magnitude only: different tokenizer; ~250 extra tokens/request for the three questions and framing.
    proxy = sum(r['qwen_tokens'] + 250 for r in rows)
    print(f'Full pilot: {len(y)} Jev requests; approx. {proxy:,} input tokens (Qwen tokenizer proxy).')
    cache = collect(rows, args.mode, OUT)
    if len(cache) != len(rows):
        print(f'Jev: {len(cache)}/{len(rows)} saved; paired result pending. Use `run` to finish.')
        return
    jev = np.array([jev_score(cache[r['id']]) for r in rows])
    tokens = [rec['input_tokens'] for rec in cache.values()]
    result = {'n': len(y), 'em_correct': int(y.sum()), 'adapted': local['adapted'], 'frozen': local['frozen'],
              'jev': measure(y, jev, original), 'paired': paired_deltas(y, scores['adapted'], jev, original),
              'requested_model': MODEL, 'reported_models': sorted({r['model'] for r in cache.values()}),
              'input_tokens': sum(tokens) if None not in tokens else None}
    save_once(OUT / 'result.json', result)
    # Publish sufficient data to replay the metrics without local hidden arrays or an API key.
    # Rounding the two probe logits makes the export insensitive to BLAS roundoff;
    # the archived-data check also reproduces ranks, both deltas and their intervals.
    save_once(OUT / 'paired_scores.json', [
        {'row': r['row'], 'id': r['id'], 'hotpot_em': r['em'],
         'adapted_probe_logit': round(r['adapted'], 10), 'frozen_probe_logit': round(r['frozen'], 10),
         'jev_accept_score': float(j)} for r, j in zip(rows, jev)
    ])
    print(json.dumps(result, indent=2))
    print(OUT / 'result.json')


if __name__ == '__main__':
    main()
