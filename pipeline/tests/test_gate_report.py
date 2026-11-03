"""The pilot gate is the only thing standing between a bad prompt and 6000 rows.

The negative case is a replay of the Gemma 4 12B incident: truncation, protocol
leaks and looping answers. It must fail on all of them.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

PIPELINE = helpers.PIPELINE
PROFILES = os.path.join(helpers.ROOT, "profiles.json")


def run_gate(run_dir, input_path, log_path, stage="pilot", profile=None):
    json_path = os.path.join(run_dir, f"gate_{stage}.json")
    args = [sys.executable, os.path.join(PIPELINE, "gate_report.py"),
            "--dir", run_dir, "--input", input_path, "--log", log_path,
            "--stage", stage, "--json", json_path]
    if profile:
        args += ["--profile", profile]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    report = None
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            report = json.load(f)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), report


def rewrite_meta(run_dir, mutate):
    meta_path = os.path.join(run_dir, "meta.jsonl")
    rows = helpers.read_jsonl(meta_path)
    for row in rows:
        mutate(row)
    helpers.write_jsonl(meta_path, rows)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def healthy(self, name="healthy", n=40):
        d = helpers.make_run(os.path.join(self.dir, name), n_per_split=n // 2)
        # keep config/meta consistent: 3 splits of n//2 rows
        inp = os.path.join(self.dir, name + ".jsonl")
        helpers.write_jsonl(inp, [
            {"id": r["id"], "split": r["split"], "question": f"Question {r['row']}?"}
            for r in helpers.read_jsonl(os.path.join(d, "meta.jsonl"))
        ])
        log = helpers.write_log(os.path.join(self.dir, name + ".log"), 3 * (n // 2),
                                truncated=0, seconds=30.0)
        return d, inp, log

    def test_healthy_pilot_passes(self):
        d, inp, log = self.healthy()
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 0, out)
        self.assertTrue(report["passed"])
        self.assertEqual(report["warns"], 0, out)
        self.assertEqual(report["summary"]["truncated"], 0)

    def test_done_summary_must_match_the_run(self):
        d, inp, log = self.healthy("wronglog")
        # Looks like a complete log, but belongs to a different-size run.
        helpers.write_log(log, rows=59, expected=60, ok=59, seconds=30.0)
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("log: processed rows match meta", out)
        self.assertIn("log: expected rows match config", out)

    def test_truncation_above_limit_fails(self):
        d, inp, log = self.healthy("trunc")
        rewrite_meta(d, lambda r: r.update(truncated=True))
        # marking every row truncated also empties the labels, which is checked too
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("truncated share", out)

    def test_main_gate_does_not_downgrade_threshold_failures(self):
        d, inp, log = self.healthy("maintrunc")
        rewrite_meta(d, lambda r: r.update(truncated=True))
        helpers.write_log(log, rows=60, expected=60, truncated=60, seconds=30.0)
        code, out, report = run_gate(d, inp, log, stage="main")
        self.assertEqual(code, 1, out)
        self.assertIn("truncated share", out)

    def test_channel_leak_fails(self):
        d, inp, log = self.healthy("leak")
        rewrite_meta(d, lambda r: r.update(answer="<|channel|>thought<channel|>hmm"))
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("no chat-protocol leak", out)

    def test_think_markers_are_reported(self):
        d, inp, log = self.healthy("think")
        rewrite_meta(d, lambda r: r.update(answer="<think>hidden reasoning</think>Paris"))
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 0, out)  # warning: some known models leak one closing marker
        self.assertGreater(report["warns"], 0)
        self.assertIn("no thinking block", out)

    def test_generate_run_requires_generated_rows(self):
        d, inp, log = self.healthy("notgenerated")

        def clear_first(row):
            if row["row"] == 0:
                row["generated"] = False
        rewrite_meta(d, clear_first)
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("all rows were generated", out)

    def test_input_id_order_must_match_meta(self):
        d, inp, log = self.healthy("wronginput")
        rows = helpers.read_jsonl(inp)
        rows[0]["id"], rows[1]["id"] = rows[1]["id"], rows[0]["id"]
        helpers.write_jsonl(inp, rows)
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("meta id order matches input", out)

    def test_looping_answers_fail(self):
        d, inp, log = self.healthy("loop")
        rewrite_meta(d, lambda r: r.update(answer="the " * 30))
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("answers are not looping", out)

    def test_skipped_rows_fail(self):
        d, inp, log = self.healthy("skip")
        rewrite_meta(d, lambda r: r.update(status="skipped", reason="too long"))
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("skipped rows", out)

    def test_shape_mismatch_fails(self):
        d, inp, log = self.healthy("shape")
        cfg_path = os.path.join(d, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["n_layers_selected"] = 2
        cfg["layers"] = [1, 2]
        helpers.write_json(cfg_path, cfg)
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)

    def test_log_estimate_is_reported(self):
        d, inp, log = self.healthy("estimate")
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 0, out)
        # 60 rows in 30 s -> 120 rows/s -> 6000 rows in 0.014 h
        self.assertLess(report["summary"]["estimated_hours_6000"], 1.0)

    def test_unaffordable_pilot_is_blocked(self):
        d, inp, log = self.healthy("slow")
        # 60 rows in 240 s = 0.25 ex/s -> 6.67 h, over the default 6 h gate.
        helpers.write_log(log, rows=60, expected=60, seconds=240.0)
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 1, out)
        self.assertIn("estimated full run is affordable", out)

    def test_historical_utf16_log_is_parsed(self):
        d, inp, log = self.healthy("utf16")
        with open(log, encoding="utf-8") as f:
            text = f.read()
        with open(log, "w", encoding="utf-16") as f:
            f.write(text)
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 0, out)
        self.assertEqual(report["warns"], 0, out)
        self.assertIsNotNone(report["summary"]["ex_per_s"])

    def test_missing_log_is_a_warning_not_a_failure(self):
        d, inp, log = self.healthy("nolog")
        os.remove(log)
        code, out, report = run_gate(d, inp, log)
        self.assertEqual(code, 0, out)
        self.assertGreater(report["warns"], 0)

    def test_lfm3_profile_raises_the_truncation_limit(self):
        if not os.path.exists(PROFILES):
            self.skipTest("profiles.json is missing")
        d, inp, log = self.healthy("lfm3", n=60)

        def truncate_every_other(row):
            if row["row"] % 2 == 0:
                row["truncated"] = True
        rewrite_meta(d, truncate_every_other)
        helpers.write_log(log, rows=90, expected=90, truncated=45, seconds=30.0)
        # 50% truncated: fails the default 10% limit, passes the LFM3 override (60%)
        code, _, _ = run_gate(d, inp, log, profile="standard")
        self.assertEqual(code, 1)
        code, _, _ = run_gate(d, inp, log, profile="lfm3")
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
