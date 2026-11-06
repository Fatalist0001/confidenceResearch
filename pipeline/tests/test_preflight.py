"""Preflight must catch the two input mistakes that already cost full runs."""

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

PIPELINE = helpers.PIPELINE


def run_preflight(args):
    proc = subprocess.run(
        [sys.executable, os.path.join(PIPELINE, "preflight.py")] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def test_clean_input_passes(self):
        path = helpers.make_input(os.path.join(self.dir, "clean.jsonl"), n=50)
        code, out = run_preflight(["--input", path, "--expect-rows", "50", "--require-instruction"])
        self.assertEqual(code, 0, out)
        self.assertIn("PASSED", out)

    def test_duplicated_instruction_fails(self):
        path = helpers.make_input(os.path.join(self.dir, "dup.jsonl"), n=50,
                                  duplicate_instruction=True)
        code, out = run_preflight(["--input", path, "--require-instruction"])
        self.assertEqual(code, 1, out)
        self.assertIn("instruction is never duplicated", out)
        self.assertIn("q_3", out)

    def test_generated_reasoning_may_quote_instruction(self):
        path = os.path.join(self.dir, "prepared.jsonl")
        instruction = "Answer with just the answer, no explanation."
        helpers.write_jsonl(path, [{
            "id": "x",
            "original_question": f"question?\n{instruction}",
            "question": f"rendered user: question?\n{instruction}\nassistant: model repeats {instruction}",
        }])
        code, out = run_preflight(["--input", path, "--require-instruction"])
        self.assertEqual(code, 0, out)

    def test_missing_instruction_is_a_failure_when_required(self):
        path = helpers.make_input(os.path.join(self.dir, "bare.jsonl"), n=20, instruction="")
        code, out = run_preflight(["--input", path, "--require-instruction"])
        self.assertEqual(code, 1, out)

    def test_empty_question_fails(self):
        path = os.path.join(self.dir, "empty.jsonl")
        helpers.write_jsonl(path, [
            {"id": "a", "question": "ok?\nAnswer with just the answer, no explanation."},
            {"id": "b", "question": "   "},
        ])
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 1, out)
        self.assertIn("no empty questions", out)

    def test_wrong_row_count_fails(self):
        path = helpers.make_input(os.path.join(self.dir, "short.jsonl"), n=10)
        code, out = run_preflight(["--input", path, "--expect-rows", "200"])
        self.assertEqual(code, 1, out)
        self.assertIn("row count", out)

    def test_unknown_duplicate_id_fails(self):
        path = os.path.join(self.dir, "dupids.jsonl")
        helpers.write_jsonl(path, [
            {"id": "x", "question": "q1?\nAnswer with just the answer, no explanation."},
            {"id": "x", "question": "q2?\nAnswer with just the answer, no explanation."},
        ])
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 1, out)

    def test_known_duplicate_id_is_accepted(self):
        path = os.path.join(self.dir, "bb.jsonl")
        helpers.write_jsonl(path, [
            {"id": "bb_956", "question": "q1?\nAnswer with just the answer, no explanation."},
            {"id": "bb_956", "question": "q2?\nAnswer with just the answer, no explanation."},
        ])
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 0, out)

    def test_known_duplicate_id_may_appear_exactly_twice(self):
        path = os.path.join(self.dir, "bb3.jsonl")
        helpers.write_jsonl(path, [
            {"id": "bb_956", "question": f"q{i}?\nAnswer with just the answer, no explanation."}
            for i in range(3)
        ])
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 1, out)

    def test_non_object_json_row_fails_cleanly(self):
        path = os.path.join(self.dir, "array.jsonl")
        helpers.write_jsonl(path, [
            {"id": "a", "question": "q?\nAnswer with just the answer, no explanation."},
            ["not", "an", "object"],
        ])
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 1, out)
        self.assertIn("every JSON row is an object", out)
        self.assertNotIn("Traceback", out)

    def test_null_question_fails_cleanly(self):
        path = os.path.join(self.dir, "nullq.jsonl")
        helpers.write_jsonl(path, [{"id": "a", "question": None}])
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 1, out)
        self.assertIn("every row has id and question", out)
        self.assertNotIn("Traceback", out)

    def test_invalid_json_line_fails(self):
        path = os.path.join(self.dir, "broken.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"id": "a", "question": "q"}\n')
            f.write("{not json}\n")
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 1, out)

    def test_invalid_utf8_fails(self):
        path = os.path.join(self.dir, "bad-utf8.jsonl")
        with open(path, "wb") as f:
            f.write(b'{"id":"a","question":"bad ' + bytes([0xFF]) + b'"}\n')
        code, out = run_preflight(["--input", path])
        self.assertEqual(code, 1, out)
        self.assertIn("invalid UTF-8", out)

    def test_repository_main_input_passes(self):
        path = os.path.join(PIPELINE, "data", "main.jsonl")
        if not os.path.exists(path):
            self.skipTest("pipeline/data/main.jsonl is not in this checkout")
        code, out = run_preflight(["--input", path, "--expect-rows", "6000",
                                   "--require-instruction"])
        self.assertEqual(code, 0, out)


if __name__ == "__main__":
    unittest.main()
