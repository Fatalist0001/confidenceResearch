"""Transfer entries must not masquerade as trained TriviaQA probe results."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers
sys.path.insert(0, os.path.join(helpers.ROOT, "scripts"))
import build_runs
import check_state


class RunsRegistryTest(unittest.TestCase):
    def test_default_label_summary_is_unchanged(self):
        rows = [{"correct_filtered_b": "1"}, {"correct_filtered_b": "0"}, {"correct_filtered_b": ""}]
        summary = build_runs.label_summary(rows, [{"truncated": True}, {}, {}])
        self.assertEqual(summary, {"rows": 3, "labeled": 2, "null": 1, "truncated": 1, "base_rate_filtered_b": 0.5})

    def test_official_label_summary_is_explicit_and_not_filtered_b(self):
        rows = [{"hotpot_em": "1", "correct_filtered_b": "0"}, {"hotpot_em": "0", "correct_filtered_b": "0"}]
        summary = build_runs.label_summary(rows, [{}, {}], "hotpot_em")
        self.assertIsNone(summary["base_rate_filtered_b"])
        self.assertEqual(summary["label_column"], "hotpot_em")
        self.assertEqual(summary["base_rate"], 0.5)
        self.assertEqual(summary["null"], 0)

    def test_transfer_registry_accepts_test_only_without_a_target_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            run = out / "main_synthetic_transfer"
            run.mkdir()
            source = Path(tmp) / "source_probe.npz"
            source.write_bytes(b"synthetic frozen source")
            meta = [{"row": i, "id": str(i), "split": "test", "truncated": False, "status": "ok"} for i in range(2)]
            helpers.write_jsonl(run / "meta.jsonl", meta)
            helpers.write_json(run / "config.json", {"n_layers_selected": 1, "n_embd": 1024,
                               "llama_cpp_commit": "785322d17", "llama_cpp_build": 12033})
            labels = [{"row": i, "id": str(i), "split": "test", "hotpot_em": str(i)} for i in range(2)]
            import csv
            with (run / "labels.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=labels[0])
                writer.writeheader()
                writer.writerows(labels)
            helpers.write_json(run / "verify.json", {"fails": 0, "warns": 0})
            helpers.write_json(run / "gate.json", {"passed": True})
            transfer = {"source_probe": {"path": str(source), "sha256": check_state.sha256(source)},
                        "protocol": {}, "frozen_feature": {"point": "C", "layer_id": 13, "target_array_index": 0,
                        "lambda": 1000.0, "fit_on_target": False, "calibration": False},
                        "accuracy_all_attempts": {}, "coverage": {}, "primary": {"n": 2}, "metrics": {},
                        "paired_deltas": {}, "retained_accuracy_50": {}, "binary_caveat": "synthetic"}
            helpers.write_json(run / "transfer.json", transfer)
            score_rows = [dict(row, analysis_included="1", frozen_probe_logit="0.3", first_token_logprob="-0.2",
                               negative_first_token_entropy="-1.0") for row in labels]
            with (run / "transfer_scores.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=score_rows[0])
                writer.writeheader()
                writer.writerows(score_rows)
            curated = {"family": "qwen3.5", "model": "synthetic", "status": "valid", "canonical": False,
                       "usable": True, "prompt_regime": "default_bos", "notes": "synthetic transfer", "probe_status": "frozen_transfer",
                       "evaluation": {"kind": "hotpotqa_frozen_transfer", "label_column": "hotpot_em", "required_splits": ["test"]}}
            from unittest.mock import patch
            with patch.object(build_runs, "OUT", str(out)), patch.object(check_state, "OUT", str(out)):
                entry = build_runs.main_entry(run.name, curated)
                self.assertIsNone(entry["probe"])
                self.assertIsNone(entry["baseline_test_auroc"])
                self.assertEqual(entry["probe_status"], "frozen_transfer")
                self.assertFalse(entry["canonical"])
                report = check_state.Report(quiet=True)
                check_state.check_registry(report, {"main_runs": {run.name: entry}})
                self.assertEqual(report.fails, 0, report.checks)
                relocated = out / "source_run/probe_model.npz"
                relocated.parent.mkdir()
                relocated.write_bytes(source.read_bytes())
                transfer["source_probe"]["path"] = str(Path(tmp) / "missing/pipeline/out/source_run/probe_model.npz")
                helpers.write_json(run / "transfer.json", transfer)
                entry = build_runs.main_entry(run.name, curated)
                report = check_state.Report(quiet=True)
                check_state.check_registry(report, {"main_runs": {run.name: entry}})
                self.assertEqual(report.fails, 0, report.checks)


if __name__ == "__main__":
    unittest.main()
