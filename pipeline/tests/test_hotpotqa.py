"""Frozen HotpotQA transfer checks use synthetic files only, without a model/GPU."""

import argparse
import copy
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

import numpy as np
import hotpotqa as hp


def dataset_row(i=0):
    return {
        "id": f"id_{i}", "question": f"Question {i}?", "answer": "Paris",
        "supporting_facts": {"title": ["Title 9"], "sent_id": [0]},
        "context": {"title": [f"Title {j}" for j in range(10)],
                    "sentences": [[f"Sentence {j}.", f"  Tail {j}!\t"] for j in range(10)]},
    }


class HotpotAnswerTest(unittest.TestCase):
    def test_official_normalization_and_whole_answer_em(self):
        self.assertTrue(hp.exact_match_score(" The PARIS! ", "Paris"))
        self.assertFalse(hp.exact_match_score("The answer is Paris.", "Paris"))
        self.assertFalse(hp.exact_match_score("Paris or London", "Paris"))
        self.assertEqual(hp.normalize_answer("A, an; the: cat\t cat."), "cat cat")
        self.assertNotEqual(hp.normalize_answer("St John’s"), hp.normalize_answer("St Johns"))

    def test_official_f1_and_yes_no_special_rule(self):
        self.assertEqual(hp.f1_score("Paris is capital", "Paris"), (0.5, 1 / 3, 1.0))
        self.assertEqual(hp.f1_score("alpha alpha beta", "alpha beta"), (0.8, 2 / 3, 1.0))
        for prediction, gold in (("yes indeed", "yes"), ("no", "noanswer"), ("no answer", "no"), ("", "Paris")):
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(hp.f1_score(prediction, gold), (0.0, 0.0, 0.0))
        self.assertEqual(hp.f1_score("YES!", "yes"), (1.0, 1.0, 1.0))

    def test_all_attempts_include_truncated_but_missing_answers_get_zero(self):
        meta = [
            {"row": 0, "id": "a", "split": "test", "status": "ok", "generated": True, "truncated": True, "answer": "Paris"},
            {"row": 1, "id": "b", "split": "test", "status": "ok", "generated": True, "answer": None},
            {"row": 2, "id": "c", "split": "test", "status": "ok", "generated": False, "answer": "Paris"},
            {"row": 3, "id": "d", "split": "test", "status": "skipped", "generated": True, "answer": "Paris"},
        ]
        gold = [{"id": r["id"], "answer": "Paris"} for r in meta]
        labels = hp.label_rows(meta, gold)
        self.assertEqual([r["hotpot_em"] for r in labels], [1, 0, 0, 1])
        self.assertEqual([r["completed"] for r in labels], [0, 0, 0, 0])
        self.assertEqual([r["completion_reason"] for r in labels],
                         ["truncated", "empty_answer", "not_generated", "state_skipped"])


class HotpotPrepareTest(unittest.TestCase):
    def test_hash_sampling_is_disjoint_order_independent_and_answer_independent(self):
        rows = [{"id": f"id_{i}", "answer": "first"} for i in range(800)]
        selected = hp.select_samples(rows)
        ranked = sorted(rows, key=lambda r: (hashlib.sha256(
            f"hotpotqa-qwen08-v1:42:{r['id']}".encode("utf-8")).hexdigest(), r["id"]))
        ids = {kind: [r["id"] for r in part] for kind, part in selected.items()}
        self.assertEqual(ids["pilot"], [r["id"] for r in ranked[:200]])
        self.assertEqual(ids["main"], [r["id"] for r in ranked[200:700]])
        self.assertFalse(set(ids["pilot"]) & set(ids["main"]))
        changed = [{"id": r["id"], "answer": "different", "question": "different"} for r in reversed(rows)]
        self.assertEqual(ids, {k: [r["id"] for r in part] for k, part in hp.select_samples(changed).items()})
        with self.assertRaisesRegex(ValueError, "unique"):
            hp.select_samples(rows + [rows[0]])

    def test_all_ten_paragraphs_and_sentence_bytes_stay_in_dataset_order(self):
        row = dataset_row()
        text = hp.render_question(row)
        blocks = [f"[{j + 1}] Title {j}\nSentence {j}.  Tail {j}!\t" for j in range(10)]
        self.assertEqual(text, hp.PROMPT_PREFIX + "\n\n".join(blocks) +
                         f"\n\nQuestion: {row['question']}\n{hp.INSTRUCTION}")
        changed = copy.deepcopy(row)
        changed["answer"] = "London"
        changed["supporting_facts"] = {"title": ["Title 0"], "sent_id": [1]}
        self.assertEqual(hp.render_question(changed), text)
        inputs, questions = hp.make_inputs([row])
        self.assertEqual(set(inputs[0]), {"id", "question", "split", "dataset"})
        self.assertEqual(questions[0]["question"], row["question"])
        self.assertEqual(questions[0]["answer"], "Paris")
        changed["context"]["title"].pop()
        with self.assertRaisesRegex(ValueError, "paragraph lists"):
            hp.render_question(changed)

    def test_short_original_context_keeps_every_supplied_paragraph(self):
        row = dataset_row()
        row["context"]["title"] = row["context"]["title"][:2]
        row["context"]["sentences"] = row["context"]["sentences"][:2]
        inputs, _ = hp.make_inputs([row])
        text = inputs[0]["question"]
        self.assertEqual(inputs[0]["id"], row["id"])
        self.assertIn("[1] Title 0\nSentence 0.  Tail 0!\t", text)
        self.assertIn("[2] Title 1\nSentence 1.  Tail 1!\t", text)
        self.assertNotIn("[3]", text)
        self.assertNotIn("Title 9", text)

    def test_full_answer_budget_is_reserved_without_replacing_questions(self):
        inputs, _ = hp.make_inputs([dataset_row()])
        original = copy.deepcopy(inputs)
        tokens = [{"id": inputs[0]["id"], "row": 0, "add_bos": False, "first_token": 1,
                   "n_prompt": 8128, "rendered_prompt": "user\n" + inputs[0]["question"] + "\nassistant"}]
        self.assertEqual(hp.check_token_counts(tokens, inputs)["max"], 8128)
        tokens[0]["n_prompt"] += 1
        with self.assertRaisesRegex(ValueError, "do not replace"):
            hp.check_token_counts(tokens, inputs)
        self.assertEqual(inputs, original)
        tokens[0]["n_prompt"] = 100
        tokens[0]["add_bos"] = True
        with self.assertRaisesRegex(ValueError, "add_bos"):
            hp.check_token_counts(tokens, inputs)

    def test_preparation_refuses_an_existing_directory_before_touching_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("existing user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                hp.prepare(argparse.Namespace(out_dir=tmp, raw_dir="missing", model="missing"))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing user data")


class HotpotRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        model = self.root / "model.gguf"
        model.write_bytes(b"synthetic model, no weights")
        source = self.root / "source"
        source.mkdir()
        self.probe = source / "probe_model.npz"
        w = np.zeros(1025)
        w[0], w[1] = -0.25, 2.0
        np.savez(self.probe, meta=np.array('{"point":"C","layer":13,"layer_id":13,"lambda":1000}'),
                 w=w, mu=np.full(1024, 2.0), sd=np.full(1024, 4.0))
        helpers.write_json(source / "config.json", {"model": str(model), "llama_cpp_commit": "3ad2230", "llama_cpp_build": 37})
        self.addCleanup(patch.stopall)
        patch.object(hp, "COUNTS", {"pilot": 2, "main": 4}).start()
        patch.object(hp, "PROBE_PATH", self.probe).start()
        patch.object(hp, "PROBE_SHA", hp.sha256(self.probe)).start()
        patch.object(hp, "N_BOOT", 20).start()
        implementation = self.root / "implementation.py"
        implementation.write_bytes(b"synthetic frozen implementation")
        evaluator = self.root / "evaluator.py"
        evaluator.write_bytes(b"synthetic pinned evaluator")
        runtime = self.root / "runtime.exe"
        runtime.write_bytes(b"synthetic frozen runtime")
        stages = {}
        for kind, n, start in (("pilot", 2, 0), ("main", 4, 2)):
            inputs, questions = hp.make_inputs([dataset_row(i + start) for i in range(n)])
            input_path, gold_path, token_path = (self.root / f"{kind}.{suffix}" for suffix in ("jsonl", "questions.jsonl", "tokens.jsonl"))
            helpers.write_jsonl(input_path, inputs)
            helpers.write_jsonl(gold_path, questions)
            tokens = [{"id": row["id"], "row": i, "n_prompt": 100 + i, "add_bos": False, "first_token": 1,
                       **({"rendered_prompt": "user\n" + row["question"] + "\nassistant"} if i == 0 else {})}
                      for i, row in enumerate(inputs)]
            helpers.write_jsonl(token_path, tokens)
            stages[kind] = {"ids": [r["id"] for r in inputs], "input": hp.fingerprint(input_path),
                            "questions": hp.fingerprint(gold_path), "tokens": hp.fingerprint(token_path)}
        self.protocol = self.root / "protocol.json"
        self.manifest = {"schema_version": 1, "kind": hp.KIND, "profile": hp.PROFILE, "flags": hp.FLAGS,
                         "source_probe": hp.fingerprint(self.probe), "source_config": hp.fingerprint(source / "config.json"),
                         "source_audit": hp.fingerprint(evaluator),
                         "model": hp.fingerprint(model), "binary": {"commit": "785322d17", "build": 12033, "runtime": [hp.fingerprint(runtime)]},
                         "evaluation": {"expected_rows": hp.COUNTS, "official_evaluator": {"source": hp.fingerprint(evaluator)},
                                        "bootstrap": {"n": 1000, "seed": 42}},
                         "dataset": {"repo": "synthetic"}, "binary_caveat": "synthetic binary caveat",
                         "implementation": [hp.fingerprint(implementation)], "stages": stages}
        helpers.write_json(self.protocol, self.manifest)
        stage = stages["main"]
        self.runner = {"profile": hp.PROFILE, "stage": "main", "flags": hp.FLAGS,
                       "input": stage["input"]["path"], "questions": stage["questions"]["path"],
                       "input_sha256": stage["input"]["sha256"].upper(), "questions_sha256": stage["questions"]["sha256"].upper(),
                       "evaluation": {"kind": hp.KIND, "protocol_sha256": hp.sha256(self.protocol).upper()}}
        helpers.write_json(self.run / "runner.json", self.runner)
        self.config = {"model": str(model), "model_name": "Qwen3.5-0.8B", "layers": [13], "n_layers": 24,
                       "n_embd": 1024, "n_layers_selected": 1, "n_examples": 4, "generate": True,
                       "template": True, "thinking": False, "max_len": 8192, "max_new_tokens": 64,
                       "sampling": {"type": "raw_argmax"}, "batch": stage["input"]["path"],
                       "llama_cpp_commit": "785322d17", "llama_cpp_build": 12033}
        helpers.write_json(self.run / "config.json", self.config)
        self.meta = [{"id": id_, "row": i, "split": "test", "dataset": "hotpotqa", "status": "ok", "generated": True,
                      "truncated": False, "answer": answer, "pos_a": 99 + i, "pos_c": 100 + i, "n_tokens": 101 + i,
                      "logprobs": [-0.2 - i / 10], "entropies": [0.8 + i / 10]}
                     for i, (id_, answer) in enumerate(zip(stage["ids"], ("Paris", "London", "London", "Paris")))]
        helpers.write_jsonl(self.run / "meta.jsonl", self.meta)
        hidden = np.ones((4, 1, 1024), dtype=np.float16)
        hidden[[0, 3], 0, 0] = 6.0
        np.save(self.run / "hidden_a.npy", hidden)
        np.save(self.run / "hidden_c.npy", hidden)
        helpers.write_json(self.run / "verify.json", {"fails": 0})
        helpers.write_json(self.run / "gate.json", {"passed": True})

    def label_args(self):
        return argparse.Namespace(dir=str(self.run), protocol=str(self.protocol), kind="main", questions=None)

    def test_real_cpp_metadata_needs_neither_question_nor_n_prompt(self):
        _, _, meta, _ = hp.check_run(self.run, self.protocol, "main")
        self.assertNotIn("question", meta[0])
        self.assertNotIn("n_prompt", meta[0])
        self.meta[0].update(pos_a=100, n_prompt=100)
        helpers.write_jsonl(self.run / "meta.jsonl", self.meta)
        with self.assertRaisesRegex(ValueError, "prompt tokens"):
            hp.check_run(self.run, self.protocol, "main")

    def test_input_hash_and_config_batch_bind_the_prompt(self):
        self.runner["input_sha256"] = "0" * 64
        helpers.write_json(self.run / "runner.json", self.runner)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            hp.check_run(self.run, self.protocol, "main")
        self.runner["input_sha256"] = self.manifest["stages"]["main"]["input"]["sha256"]
        helpers.write_json(self.run / "runner.json", self.runner)
        self.config["batch"] = str(self.root / "another_input.jsonl")
        helpers.write_json(self.run / "config.json", self.config)
        with self.assertRaisesRegex(ValueError, "config batch"):
            hp.check_run(self.run, self.protocol, "main")

    def test_source_probe_change_is_rejected(self):
        self.probe.write_bytes(b"different weights")
        with self.assertRaisesRegex(ValueError, "changed since protocol freeze"):
            hp.check_protocol(self.protocol, "main")

    def test_source_probe_and_scoring_code_cannot_change_after_freeze(self):
        implementation = Path(self.manifest["implementation"][0]["path"])
        implementation.write_bytes(b"changed implementation")
        with self.assertRaisesRegex(ValueError, "changed since protocol freeze"):
            hp.check_protocol(self.protocol, "main")

    def test_sampling_and_layer_changes_are_rejected(self):
        self.config["sampling"]["type"] = "common_sampler"
        helpers.write_json(self.run / "config.json", self.config)
        with self.assertRaisesRegex(ValueError, "raw-argmax"):
            hp.check_run(self.run, self.protocol, "main")
        self.config["sampling"]["type"] = "raw_argmax"
        self.config["layers"] = [12]
        helpers.write_json(self.run / "config.json", self.config)
        with self.assertRaisesRegex(ValueError, "model-layer-13"):
            hp.check_run(self.run, self.protocol, "main")

    def test_primary_mask_is_shared_and_requires_valid_c_and_first_token_scores(self):
        hidden = np.ones((4, 1, 1024), dtype=np.float16)
        self.meta[1]["entropies"] = []
        hidden[2] = 0
        self.meta[3]["logprobs"] = [float("nan")]
        idx, reasons = hp.primary_indices(self.meta, hidden, 0)
        self.assertEqual(idx.tolist(), [0])
        self.assertEqual(reasons, ["included", "missing_first_token_scores", "invalid_c_state", "invalid_first_token_scores"])

    def test_label_and_score_are_create_only_and_never_fit(self):
        self.meta[1]["truncated"] = True
        self.meta[3]["answer"] = ""
        helpers.write_jsonl(self.run / "meta.jsonl", self.meta)
        before = {p: hp.sha256(p) for p in (self.probe, self.run / "meta.jsonl", self.run / "hidden_c.npy")}
        hp.label(self.label_args())
        label_hash = hp.sha256(self.run / "labels.csv")
        with self.assertRaises(FileExistsError):
            hp.label(self.label_args())
        out, scores = self.run / "transfer.json", self.run / "transfer_scores.csv"
        args = argparse.Namespace(dir=str(self.run), protocol=str(self.protocol), out=str(out), scores_out=str(scores))
        with patch("baseline.fit_logistic", side_effect=AssertionError("target calibration is forbidden")):
            hp.score(args)
        result = hp.read_json(out)
        self.assertEqual(result["accuracy_all_attempts"]["n"], 4)
        self.assertEqual(result["primary"]["n"], 2)
        self.assertEqual(result["frozen_feature"]["target_array_index"], 0)
        self.assertFalse(result["frozen_feature"]["fit_on_target"])
        self.assertEqual({r["n"] for r in result["metrics"].values()}, {2})
        rows = helpers.read_csv_rows(scores)
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["analysis_included"] for r in rows], ["1", "0", "1", "0"])
        self.assertAlmostEqual(float(rows[0]["frozen_probe_logit"]), 1.75)
        self.assertAlmostEqual(float(rows[2]["frozen_probe_logit"]), -0.75)
        self.assertEqual(label_hash, hp.sha256(self.run / "labels.csv"))
        self.assertEqual(before, {p: hp.sha256(p) for p in before})
        outputs = {p: hp.sha256(p) for p in (out, scores)}
        with self.assertRaises(FileExistsError):
            hp.score(args)
        self.assertEqual(outputs, {p: hp.sha256(p) for p in outputs})

    def test_failed_verify_blocks_scoring_without_creating_outputs(self):
        helpers.write_json(self.run / "verify.json", {"fails": 1})
        out, scores = self.run / "transfer.json", self.run / "transfer_scores.csv"
        with self.assertRaisesRegex(ValueError, "verify failed"):
            hp.score(argparse.Namespace(dir=str(self.run), protocol=str(self.protocol), out=str(out), scores_out=str(scores)))
        self.assertFalse(out.exists())
        self.assertFalse(scores.exists())


class HotpotComparisonTest(unittest.TestCase):
    def test_identical_scores_have_zero_paired_bootstrap_differences(self):
        y = np.asarray([1, 0, 1, 0], dtype=float)
        score = np.asarray([0.9, 0.5, 0.5, 0.1])
        result = hp.paired_delta(y, score, score, n_boot=100)
        for metric in result.values():
            self.assertEqual(metric, {"delta": 0.0, "ci": [0.0, 0.0]})

    def test_retained_half_uses_ceil_and_original_row_order_for_ties(self):
        result = hp.retained_accuracy(np.asarray([1, 0, 1, 0, 0]), np.ones(5))
        self.assertEqual(result["retained"], 3)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
