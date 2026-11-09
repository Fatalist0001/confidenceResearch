"""verify_run.py must accept a consistent directory and reject the known defects."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

import numpy as np

PIPELINE = helpers.PIPELINE


def run_verify(run_dir, extra=None):
    json_path = os.path.join(run_dir, "verify.json")
    args = [sys.executable, os.path.join(PIPELINE, "verify_run.py"),
            "--dir", run_dir, "--json", json_path, "--full"] + (extra or [])
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    report = None
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            report = json.load(f)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), report


class VerifyRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_consistent_run_passes(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "good"))
        code, out, report = run_verify(d)
        self.assertEqual(code, 0, out)
        self.assertEqual(report["fails"], 0, out)

    def test_wrong_array_shape_fails(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "shape"))
        arr = np.load(os.path.join(d, "hidden_a.npy"))
        np.save(os.path.join(d, "hidden_a.npy"), arr[:-1])
        code, out, report = run_verify(d)
        self.assertEqual(code, 1, out)
        self.assertGreater(report["fails"], 0)

    def test_missing_file_still_writes_failure_report(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "missing"))
        os.remove(os.path.join(d, "hidden_c.npy"))
        code, out, report = run_verify(d)
        self.assertEqual(code, 1, out)
        self.assertIsNotNone(report)
        self.assertGreater(report["fails"], 0)

    def test_all_zero_layer_fails(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "zeros"))
        arr = np.load(os.path.join(d, "hidden_a.npy"))
        arr[:, 2, :] = 0
        np.save(os.path.join(d, "hidden_a.npy"), arr)
        code, out, report = run_verify(d)
        self.assertEqual(code, 1, out)
        self.assertIn("every layer has non-zero values", out)

    def test_all_zero_ok_row_fails(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "zerorow"))
        arr = np.load(os.path.join(d, "hidden_c.npy"))
        arr[0, :, :] = 0
        np.save(os.path.join(d, "hidden_c.npy"), arr)
        code, out, _ = run_verify(d)
        self.assertEqual(code, 1, out)

    def test_meta_row_gap_fails(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "gap"))
        meta_path = os.path.join(d, "meta.jsonl")
        with open(meta_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        with open(meta_path, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])
        code, out, _ = run_verify(d)
        self.assertEqual(code, 1, out)

    def test_single_class_labels_fail(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "oneclass"))
        labels_path = os.path.join(d, "labels.csv")
        with open(labels_path, encoding="utf-8") as f:
            text = f.read()
        lines = []
        for line in text.splitlines():
            parts = line.split(",")
            if len(parts) == 4 and parts[3] == "1":
                parts[3] = "0"
            lines.append(",".join(parts))
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        code, out, _ = run_verify(d)
        self.assertEqual(code, 1, out)
        self.assertIn("both classes present", out)

    def test_single_class_in_one_split_fails(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "onesplit"))
        labels_path = os.path.join(d, "labels.csv")
        rows = helpers.read_csv_rows(labels_path)
        for row in rows:
            if row["split"] == "test":
                row["correct_filtered_b"] = "0"
        import csv
        with open(labels_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        code, out, _ = run_verify(d)
        self.assertEqual(code, 1, out)
        self.assertIn("both classes in every split", out)

    def test_missing_labels_is_only_a_warning(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "nolabels"), labels=False)
        code, out, report = run_verify(d)
        self.assertEqual(code, 0, out)
        self.assertGreater(report["warns"], 0)

    def test_split_outside_train_val_test_fails(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "split"))
        meta_path = os.path.join(d, "meta.jsonl")
        rows = helpers.read_jsonl(meta_path)
        rows[0]["split"] = "dev"
        helpers.write_jsonl(meta_path, rows)
        code, out, _ = run_verify(d)
        self.assertEqual(code, 1, out)
        self.assertIn("all rows have a split", out)

    def test_official_test_only_labels_need_explicit_evaluation_mode(self):
        d = helpers.make_run(os.path.join(self.tmp.name, "transfer"))
        meta_path = os.path.join(d, "meta.jsonl")
        meta = helpers.read_jsonl(meta_path)
        for row in meta:
            row["split"] = "test"
        helpers.write_jsonl(meta_path, meta)
        labels_path = os.path.join(d, "labels.csv")
        labels = helpers.read_csv_rows(labels_path)
        for row in labels:
            row["split"] = "test"
            row["hotpot_em"] = row.pop("correct_filtered_b")
        import csv
        with open(labels_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=labels[0].keys())
            writer.writeheader()
            writer.writerows(labels)
        code, out, report = run_verify(d, ["--label-column", "hotpot_em", "--required-splits", "test"])
        self.assertEqual(code, 0, out)
        self.assertEqual(report["fails"], 0)
        code, out, _ = run_verify(d, ["--label-column", "hotpot_em"])
        self.assertEqual(code, 1, out)
        self.assertIn("train/val/test all non-empty", out)


if __name__ == "__main__":
    unittest.main()
