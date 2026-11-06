import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm_reasoning import (  # noqa: E402
    build_final_prompts,
    extract_first_final_line,
    finalize_replay,
    make_remaining,
    merge_reasoning,
    read_meta_rows,
    select_budget,
    write_jsonl,
)


class LfmReasoningTest(unittest.TestCase):
    def test_build_prompt_cuts_natural_final_and_keeps_one_instruction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions = root / "questions.jsonl"
            meta = root / "meta.jsonl"
            out = root / "out.jsonl"
            write_jsonl(
                questions,
                [{"id": "x", "question": "Q?\nAnswer with just the answer, no explanation.", "split": "train"}],
            )
            write_jsonl(
                meta,
                [{"id": "x", "status": "ok", "generated": True, "truncated": False,
                  "answer": "<think>reason</think>old final", "pos_a": 10, "pos_c": 13}],
            )
            summary = build_final_prompts(questions, meta, out, 32)
            row = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(summary["natural_closes"], 1)
            self.assertEqual(row["reasoning"], "<think>reason")
            self.assertTrue(row["question"].endswith("<think>reason\n</think>\nFinal answer:"))
            self.assertEqual(row["question"].count("Answer with just the answer, no explanation."), 1)

    def test_reasoning_may_quote_the_instruction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions = root / "questions.jsonl"
            meta = root / "meta.jsonl"
            out = root / "out.jsonl"
            instruction = "Answer with just the answer, no explanation."
            write_jsonl(questions, [{"id": "x", "question": f"Q?\n{instruction}"}])
            write_jsonl(
                meta,
                [{"id": "x", "status": "ok", "generated": True, "truncated": True,
                  "answer": f"<think>The user said {instruction}"}],
            )
            build_final_prompts(questions, meta, out, 32)
            row = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(row["question"].count(instruction), 2)

    def test_final_extraction_requires_a_stop_and_rejects_reopened_thinking(self):
        self.assertEqual(extract_first_final_line(" answer\nmore", True), (" answer", "newline"))
        self.assertEqual(extract_first_final_line(" answer", False), (" answer", "eog"))
        self.assertEqual(extract_first_final_line(" answer</think>more", True), (" answer", "marker"))
        self.assertEqual(
            extract_first_final_line(" The answer is X. Wait, reconsider.", True),
            (" The answer is X.", "reasoning_restart"),
        )
        self.assertEqual(extract_first_final_line(" Wait Until Dark", True), ("", "no_final_stop"))
        self.assertEqual(extract_first_final_line(" Tom Waits\n", True), (" Tom Waits", "newline"))
        self.assertEqual(extract_first_final_line(" answer", True), ("", "no_final_stop"))
        self.assertEqual(extract_first_final_line("<think>again\n", True), ("", "reasoning_reopened"))

    def test_selects_smallest_near_best_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidates = []
            for budget, positives, coverage in [(32, 75, 0.94), (64, 80, 0.98), (128, 82, 1.0)]:
                labels = root / f"labels_{budget}.csv"
                with open(labels, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["correct_filtered_b"])
                    writer.writeheader()
                    for i in range(100):
                        writer.writerow({"correct_filtered_b": "1" if i < positives else "0"})
                summary = root / f"summary_{budget}.json"
                summary.write_text(
                    json.dumps({"rows": 100, "extracted": round(100 * coverage), "coverage": coverage}),
                    encoding="utf-8",
                )
                candidates.append((budget, labels, summary))
            result = select_budget(candidates, root / "selection.json", 0.95, 0.02)
            self.assertEqual(result["selected_budget"], 64)

    def test_finalize_replay_restores_generation_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.jsonl"
            out = root / "meta.jsonl"
            write_jsonl(
                raw,
                [{"id": "x", "answer": " final", "generated": False, "truncated": False,
                  "final_generated": True, "final_logprobs": [-0.2], "final_entropies": [0.3]}],
            )
            summary = finalize_replay(raw, out)
            row = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(summary, {"rows": 1, "extraction_failures": 0})
            self.assertTrue(row["generated"])
            self.assertEqual(row["logprobs"], [-0.2])
            self.assertEqual(row["entropies"], [0.3])
            self.assertEqual(row["generation_protocol"], "lfm25-thinking-final-v1")

    def test_merge_restores_global_rows_and_tolerates_a_cut_tail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions = root / "questions.jsonl"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            out = root / "merged.jsonl"
            write_jsonl(questions, [{"id": f"q{i}"} for i in range(5)])
            write_jsonl(
                first,
                [{"id": "q0", "row": 0, "status": "ok", "generated": True},
                 {"id": "q1", "row": 1, "status": "ok", "generated": True}],
            )
            first.write_text(first.read_text(encoding="utf-8") + '{"id":"q2","row":2,"sta', encoding="utf-8")
            write_jsonl(
                second,
                [{"id": "q2", "row": 0, "status": "ok", "generated": True},
                 {"id": "q3", "row": 1, "status": "ok", "generated": True},
                 {"id": "q4", "row": 2, "status": "ok", "generated": True}],
            )
            self.assertEqual(len(read_meta_rows(first)), 2)
            summary = merge_reasoning(questions, [first, second], out)
            merged = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["rows"], 5)
            self.assertEqual([row["row"] for row in merged], [0, 1, 2, 3, 4])
            self.assertEqual([row["id"] for row in merged], ["q0", "q1", "q2", "q3", "q4"])

    def test_merge_rejects_a_gap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions = root / "questions.jsonl"
            part = root / "part.jsonl"
            write_jsonl(questions, [{"id": f"q{i}"} for i in range(3)])
            write_jsonl(
                part,
                [{"id": "q0", "row": 0, "status": "ok", "generated": True},
                 {"id": "q2", "row": 1, "status": "ok", "generated": True}],
            )
            with self.assertRaises(ValueError):
                merge_reasoning(questions, [part], root / "merged.jsonl")

    def test_make_remaining_writes_the_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.jsonl"
            out = root / "rest.jsonl"
            write_jsonl(source, [{"id": f"q{i}"} for i in range(5)])
            summary = make_remaining(source, out, 3)
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["first_id"], "q3")
            self.assertEqual([json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()],
                             ["q3", "q4"])


if __name__ == "__main__":
    unittest.main()
