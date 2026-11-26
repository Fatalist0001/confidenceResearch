"""Audit the published comparison using only versioned data: no model, API or .npy files."""

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
from jev_hotpot_pilot import jev_score, measure, paired_deltas

QUALITY = PIPELINE / 'out/jev_hotpot_pilot_v1'
TIME = PIPELINE / 'out/latency'


class JevReportTests(unittest.TestCase):
    def test_quality_replays_from_independently_joined_labels_and_api_answers(self):
        result = json.loads((QUALITY / 'result.json').read_bytes())
        proto = json.loads((QUALITY / 'protocol.json').read_bytes())
        rows = json.loads((QUALITY / 'paired_scores.json').read_bytes())
        raw = [json.loads(line) for line in (QUALITY / 'jev_scores.jsonl').read_bytes().splitlines()]
        api = {r['id']: r for r in raw}
        cohort = PIPELINE / 'out/main_qwen08_q4_hotpotqa_v2/transfer_scores.csv'
        with cohort.open(encoding='utf-8', newline='') as f:
            labels = {r['id']: r for r in csv.DictReader(f) if r['analysis_included'] == '1'}
        ids = [r['id'] for r in rows]
        train = set(proto['fit']['train_ids'])
        self.assertEqual(ids, [r['id'] for r in proto['test']])
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(api), len(raw))
        self.assertEqual(set(ids), set(api))
        self.assertTrue(train.isdisjoint(ids))
        self.assertEqual(train.union(ids), set(labels))
        self.assertEqual({r['model'] for r in raw}, {result['requested_model']})
        self.assertEqual(sum(r['input_tokens'] for r in raw), result['input_tokens'])
        for r, request in zip(rows, proto['test']):
            qid = r['id']
            self.assertEqual(r['row'], request['row'])
            self.assertEqual(r['row'], int(labels[qid]['row']))
            self.assertEqual(r['hotpot_em'], int(labels[qid]['hotpot_em']))
            self.assertAlmostEqual(r['frozen_probe_logit'], float(labels[qid]['frozen_probe_logit']), places=9)
            self.assertEqual(api[qid]['request_sha256'], request['request_sha256'])
            self.assertEqual(r['jev_accept_score'], jev_score(api[qid]))
        y, idx, adapted, frozen, jev = (np.array([r[k] for r in rows]) for k in
            ('hotpot_em', 'row', 'adapted_probe_logit', 'frozen_probe_logit', 'jev_accept_score'))
        self.assertEqual((len(y), int(y.sum())), (result['n'], result['em_correct']))
        for name, score in [('adapted', adapted), ('frozen', frozen), ('jev', jev)]:
            self.assertEqual(measure(y, score, idx), result[name])
        for name, value in paired_deltas(y, adapted, jev, idx).items():
            np.testing.assert_allclose(value, result['paired'][name], rtol=0, atol=1e-12)

    def test_latency_preserves_failure_and_counts_each_success_only_once(self):
        final = json.loads((TIME / 'jev_20261126T125654641493Z.json').read_bytes())
        probe = json.loads((TIME / 'probe_20261126T122838740827Z.json').read_bytes())
        parent = (TIME / final['resumed_from']['file']).read_bytes()
        self.assertEqual(hashlib.sha256(parent).hexdigest(), final['resumed_from']['sha256'])
        self.assertEqual(final['attempts'][:len(json.loads(parent)['attempts'])],
                         [dict(r, session=1) for r in json.loads(parent)['attempts']])
        attempts = [r for r in final['attempts'] if r['phase'] == 'measure']
        good = [r for r in attempts if r['ok']]
        self.assertTrue(final['complete'])
        self.assertEqual(final['failed_attempts'], sum(not r['ok'] for r in final['attempts']))
        self.assertEqual(len(good), len({r['id'] for r in good}))
        self.assertEqual(len(good), final['planned_measured'])
        self.assertEqual([r['id'] for r in good], [r['id'] for r in probe['per_input']])
        main_ids = {r['id'] for r in json.loads((QUALITY / 'paired_scores.json').read_bytes())}
        self.assertTrue(main_ids.isdisjoint(r['id'] for r in good))
        first = {}
        for r in attempts:
            first.setdefault(r['id'], r)
        for key, group in [('post_warmup_total_ms', good),
                           ('first_attempt_success_total_ms', [r for r in first.values() if r['ok']]),
                           ('reused_connection_total_ms', [r for r in good if r['new_connections'] == 0])]:
            values = [r['total_ms'] for r in group]
            recorded = final[key]
            self.assertEqual(recorded['n'], len(values))
            np.testing.assert_allclose([recorded['p50'], recorded['p95']], np.percentile(values, [50, 95]))
