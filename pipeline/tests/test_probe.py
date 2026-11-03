"""The frozen probe protocol: val-only selection, one test evaluation, determinism.

The synthetic data carries the label in layer index 1, so a correct run must pick
that layer; with random labels it must not look better than chance.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

PIPELINE = helpers.PIPELINE


def run_probe(run_dir, prefix="probe", model_out=True, extra=None):
    args = [
        sys.executable, os.path.join(PIPELINE, "probe.py"),
        "--hidden-a", os.path.join(run_dir, "hidden_a.npy"),
        "--hidden-c", os.path.join(run_dir, "hidden_c.npy"),
        "--meta", os.path.join(run_dir, "meta.jsonl"),
        "--labels", os.path.join(run_dir, "labels.csv"),
        "--out", os.path.join(run_dir, f"{prefix}.json"),
        "--curve-out", os.path.join(run_dir, f"{prefix}_curve.csv"),
        "--scores-out", os.path.join(run_dir, f"{prefix}_scores.csv"),
    ]
    if model_out:
        args += ["--model-out", os.path.join(run_dir, f"{prefix}_model.npz")]
    args += extra or ["--n-boot", "200", "--shuffles", "3"]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out_path = os.path.join(run_dir, f"{prefix}.json")
    report = None
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            report = json.load(f)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), report


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def test_signal_is_recovered(self):
        d = helpers.make_run(os.path.join(self.dir, "signal"), n_per_split=60, seed=1)
        code, out, report = run_probe(d)
        self.assertEqual(code, 0, out)
        self.assertGreater(report["metrics"]["test"]["auroc"], 0.9, out)
        # layer 1 (the second selected layer) carries the signal
        self.assertEqual(report["selected"]["layer"], 1, out)

    def test_noise_stays_near_chance(self):
        d = helpers.make_run(os.path.join(self.dir, "noise"), n_per_split=60, seed=2, signal=False)
        code, out, report = run_probe(d)
        self.assertEqual(code, 0, out)
        self.assertLess(report["metrics"]["test"]["auroc"], 0.75, out)

    def test_repeat_is_byte_identical(self):
        d = helpers.make_run(os.path.join(self.dir, "det"), n_per_split=40, seed=3)
        code1, out1, _ = run_probe(d, prefix="probe")
        code2, out2, _ = run_probe(d, prefix="probe2")
        self.assertEqual((code1, code2), (0, 0), out1 + out2)
        for name in ("probe_curve.csv", "probe_scores.csv"):
            self.assertEqual(
                sha256(os.path.join(d, name)),
                sha256(os.path.join(d, name.replace("probe", "probe2", 1))),
                name,
            )

    def test_selection_uses_val_not_test(self):
        d = helpers.make_run(os.path.join(self.dir, "valonly"), n_per_split=50, seed=4)
        code, out, report = run_probe(d)
        self.assertEqual(code, 0, out)
        # the lambda is chosen on the mean val AUROC across the whole grid, and the
        # point/layer afterwards on val AUROC within that lambda (probe.py:237-246)
        curve = helpers.read_csv_rows(os.path.join(d, "probe_curve.csv"))
        self.assertTrue(curve)
        means = {}
        for row in curve:
            means.setdefault(float(row["lambda"]), []).append(float(row["val_auroc"]))
        best = max(means.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
        self.assertEqual(report["selected_lambda"], best)
        cand = [r for r in curve if float(r["lambda"]) == best]
        winner = max(cand, key=lambda r: float(r["val_auroc"]))
        self.assertEqual(report["selected"]["point"], winner["point"])
        self.assertEqual(report["selected"]["layer"], int(winner["layer"]))

    def test_scores_cover_val_and_test_only(self):
        d = helpers.make_run(os.path.join(self.dir, "scores"), n_per_split=40, seed=5)
        code, out, report = run_probe(d)
        self.assertEqual(code, 0, out)
        rows = helpers.read_csv_rows(os.path.join(d, "probe_scores.csv"))
        splits = {r["split"] for r in rows}
        self.assertEqual(splits, {"val", "test"})
        self.assertEqual(len(rows), report["rows"]["val"] + report["rows"]["test"])

    def test_probe_model_roundtrip(self):
        d = helpers.make_run(os.path.join(self.dir, "npz"), n_per_split=40, seed=6)
        code, out, report = run_probe(d)
        self.assertEqual(code, 0, out)
        import numpy as np
        with np.load(os.path.join(d, "probe_model.npz"), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            self.assertEqual(len(z["w"]), len(z["mu"]) + 1)
            self.assertEqual(len(z["mu"]), len(z["sd"]))
            self.assertEqual(meta["layer"], report["selected"]["layer"])
            self.assertEqual(float(meta["lambda"]), float(report["selected_lambda"]))

    def test_missing_split_is_an_error(self):
        d = helpers.make_run(os.path.join(self.dir, "nosplit"), n_per_split=10, seed=7)
        meta_path = os.path.join(d, "meta.jsonl")
        rows = helpers.read_jsonl(meta_path)
        for row in rows:
            if row["split"] == "test":
                row["split"] = "val"
        helpers.write_jsonl(meta_path, rows)
        labels_path = os.path.join(d, "labels.csv")
        with open(labels_path, encoding="utf-8") as f:
            text = f.read().replace(",test,", ",val,")
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write(text)
        code, out, report = run_probe(d)
        self.assertNotEqual(code, 0, out)
        self.assertIn("no usable rows for split 'test'", out)


if __name__ == "__main__":
    unittest.main()
