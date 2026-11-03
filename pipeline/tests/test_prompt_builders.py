"""Prepared prompt files must preserve metadata and never duplicate instructions."""

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers


INSTRUCTION = "Answer with just the answer, no explanation."


def run_builder(script, src, dst):
    proc = subprocess.run(
        [sys.executable, os.path.join(helpers.PIPELINE, script), "--in", src, "--out", dst],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


class PromptBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "input.jsonl")
        helpers.write_jsonl(self.src, [{
            "id": "q1",
            "question": "Who wrote Hamlet?\n" + INSTRUCTION,
            "gold": ["William Shakespeare"],
            "split": "train",
            "custom": {"keep": True},
        }])

    def test_gptoss_prompt_is_deterministic_and_has_one_instruction(self):
        a = os.path.join(self.tmp.name, "gpt-a.jsonl")
        b = os.path.join(self.tmp.name, "gpt-b.jsonl")
        code_a, out_a = run_builder("make_gptoss_prompt.py", self.src, a)
        code_b, out_b = run_builder("make_gptoss_prompt.py", self.src, b)
        self.assertEqual((code_a, code_b), (0, 0), out_a + out_b)
        self.assertEqual(read_bytes(a), read_bytes(b))
        row = helpers.read_jsonl(a)[0]
        self.assertEqual(row["question"].count(INSTRUCTION), 1)
        self.assertIn("<|channel|>final<|message|>", row["question"])
        self.assertEqual(row["custom"], {"keep": True})

    def test_lfm_prompt_is_self_idempotent(self):
        a = os.path.join(self.tmp.name, "lfm-a.jsonl")
        b = os.path.join(self.tmp.name, "lfm-b.jsonl")
        code_a, out_a = run_builder("make_lfm_final_prompt.py", self.src, a)
        code_b, out_b = run_builder("make_lfm_final_prompt.py", a, b)
        self.assertEqual((code_a, code_b), (0, 0), out_a + out_b)
        self.assertEqual(read_bytes(a), read_bytes(b))
        row = helpers.read_jsonl(a)[0]
        self.assertEqual(row["question"].count(INSTRUCTION), 1)
        self.assertTrue(row["question"].endswith("<think></think>\nFinal answer:"))
        self.assertEqual(row["gold"], ["William Shakespeare"])


if __name__ == "__main__":
    unittest.main()
