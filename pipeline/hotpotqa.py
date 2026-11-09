"""Prepare and evaluate the Qwen 0.8B frozen-probe HotpotQA transfer test.

No fitting, calibration, retrieval, answer extraction or layer selection lives here.
The public parquet and official evaluator are pinned; every output is create-only.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import string
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from apply_probe_model import load_probe_model, score_hidden
from baseline import auroc, auprc, ci, metrics, norm_path
from preflight import INSTRUCTION


ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
PROFILE = "qwen-hotpotqa"
KIND = "hotpotqa_frozen_transfer"
COUNTS = {"pilot": 200, "main": 500}
SEED = 42
N_BOOT = 1000
DATA_REV = "1908d6afbbead072334abe2965f91bd2709910ab"
DATA_FILE = "validation-00000-of-00001.parquet"
DATA_SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
EVAL_REV = "fa3a36370899e1d85822de61e58c85ea19993154"
EVAL_FILE = "hotpot_evaluate_v1.fa3a363.py"
EVAL_SHA = "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0"
PROBE_PATH = PIPELINE / "out/main_qwen08_q4/probe_model.npz"
PROBE_SHA = "5e948cd2146d95fef7ad2b66b21c9c5d12fdd3cfc02c250f30fcb81a1ca7088c"
SOURCE_AUDIT_PATH = PIPELINE / "data/hotpotqa_source_audit_v1.json"
SAMPLE_PREFIX = "hotpotqa-qwen08-v1"
PROMPT_PREFIX = "Use the supplied paragraphs to answer the question.\n\nParagraphs:\n"
FLAGS = {
    "generate": True, "max_new_tokens": 64, "max_len": 8192, "layers": "13",
    "ngl": 99, "no_template": False, "no_thinking": False, "thinking": False,
    "batch_rows": 1, "temperature": None, "top_k": None,
    "repeat_penalty": None, "seed": None,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": norm_path(str(path)), "sha256": sha256(path), "bytes": path.stat().st_size}


def check_fingerprint(record, path=None):
    path = Path(path or record["path"]).resolve()
    require(path == Path(record["path"]).resolve(), f"wrong source path: {path}")
    require(path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"],
            f"file changed since protocol freeze: {path}")
    return path


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_absent(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"output already exists: {path}; choose a new filename/tag")


def write_json(path, data):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path, rows):
    with open(path, "x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize_answer(text):
    # Official hotpot_evaluate_v1.py, pinned above: ASCII punctuation only.
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction, gold):
    return normalize_answer(prediction) == normalize_answer(gold)


def f1_score(prediction, gold):
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
        return 0.0, 0.0, 0.0
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    if not common:
        return 0.0, 0.0, 0.0
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall), precision, recall


def select_samples(rows, pilot_n=200, main_n=500, seed=SEED):
    ids = [r["id"] for r in rows]
    require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids),
            "dataset IDs must be non-empty and unique")
    require(len(rows) >= pilot_n + main_n, "not enough rows for disjoint pilot and evaluation")
    ranked = sorted(rows, key=lambda r: (
        hashlib.sha256(f"{SAMPLE_PREFIX}:{seed}:{r['id']}".encode("utf-8")).hexdigest(), r["id"]))
    return {"pilot": ranked[:pilot_n], "main": ranked[pilot_n:pilot_n + main_n]}


def render_question(row):
    context = row["context"]
    titles, paragraphs = context["title"], context["sentences"]
    require(titles and len(titles) == len(paragraphs), f"{row['id']}: malformed source paragraph lists")
    blocks = []
    for i, (title, sentences) in enumerate(zip(titles, paragraphs), 1):
        require(isinstance(title, str) and sentences and all(isinstance(s, str) for s in sentences),
                f"{row['id']}: malformed paragraph {i}")
        blocks.append(f"[{i}] {title}\n{''.join(sentences)}")
    require(isinstance(row["question"], str) and row["question"].strip(), "empty question")
    return PROMPT_PREFIX + "\n\n".join(blocks) + f"\n\nQuestion: {row['question']}\n{INSTRUCTION}"


def make_inputs(rows):
    inputs, questions = [], []
    for row in rows:
        inputs.append({"id": row["id"], "question": render_question(row), "split": "test", "dataset": "hotpotqa"})
        require(isinstance(row["answer"], str) and normalize_answer(row["answer"]), "empty gold answer")
        questions.append({"id": row["id"], "question": row["question"], "answer": row["answer"], "split": "test"})
    return inputs, questions


def check_token_counts(tokens, inputs, max_len=8192, answer_budget=64):
    require([r.get("id") for r in tokens] == [r["id"] for r in inputs], "token diagnostic ID order mismatch")
    require([r.get("row") for r in tokens] == list(range(len(inputs))), "token diagnostic row mismatch")
    require(all(r.get("add_bos") is False for r in tokens), "Qwen must report add_bos=false")
    lengths = [r["n_prompt"] for r in tokens]
    require(lengths and all(isinstance(n, int) and 0 < n <= max_len - answer_budget for n in lengths),
            "a prompt does not leave the frozen 64-token answer budget; do not replace questions")
    require(all(r["first_token"] == tokens[0]["first_token"] for r in tokens), "inconsistent first prompt token")
    require(tokens[0]["rendered_prompt"].count(inputs[0]["question"]) == 1, "rendered prompt changed/doubled the user text")
    return {"rows": len(tokens), "min": min(lengths), "max": max(lengths),
            "median": float(np.median(lengths)), "answer_budget": answer_budget,
            "first_token": tokens[0]["first_token"], "add_bos": False,
            "first_rendered_sha256": hashlib.sha256(tokens[0]["rendered_prompt"].encode("utf-8")).hexdigest()}


def profile_flags():
    profiles = read_json(ROOT / "profiles.json")
    flags = {"thinking": False, **profiles["defaults"]["flags"], **profiles["profiles"][PROFILE]["flags"]}
    require(flags == FLAGS, "HotpotQA profile changed from the frozen execution flags")
    return flags


def prepare(args):
    import pyarrow.parquet as pq

    out = Path(args.out_dir).resolve()
    assert_absent(out)
    raw = Path(args.raw_dir).resolve()
    require(sha256(raw / DATA_FILE) == DATA_SHA and (raw / DATA_FILE).stat().st_size == 27452575,
            "wrong pinned HotpotQA parquet")
    require(sha256(raw / EVAL_FILE) == EVAL_SHA, "wrong pinned official evaluator")
    require(sha256(PROBE_PATH) == PROBE_SHA, "canonical Qwen source probe changed")
    probe = load_probe_model(PROBE_PATH)
    require((probe["info"]["point"], probe["info"]["layer_id"], probe["info"]["lambda"], len(probe["mu"])) ==
            ("C", 13, 1000.0, 1024), "wrong frozen probe configuration")
    source_config = read_json(PROBE_PATH.parent / "config.json")
    require(Path(args.model).resolve() == Path(source_config["model"]).resolve(), "use the source probe's Qwen 0.8B Q4 model")
    active = read_json(ROOT / "PROJECT_STATE.json")["binaries"]["active"]
    exe, dump = ROOT / active["exe"], ROOT / active["prompt_dump"]
    version = subprocess.run([str(exe), "--version"], capture_output=True, check=True, text=True, encoding="utf-8")
    require(f"build {active['build']}, commit {active['commit']}" in version.stdout + version.stderr, "active binary version mismatch")
    flags = profile_flags()
    table = pq.read_table(raw / DATA_FILE, columns=["id", "question", "context", "answer"])
    require(table.num_rows == 7405, "wrong HotpotQA dev row count")
    selected = select_samples(table.to_pylist())
    audit = read_json(SOURCE_AUDIT_PATH)
    require(audit["parquet"]["sha256"] == DATA_SHA and audit["comparison"]["same_id_order"] is True and
            not any(audit["comparison"]["differences_by_field"].values()), "source audit does not verify this parquet")
    require(all([r["id"] for r in rows] == audit["stages"][kind]["ids"] for kind, rows in selected.items()),
            "sample IDs differ from the source audit; do not replace questions")
    model = fingerprint(args.model)
    runtime = [fingerprint(exe), fingerprint(dump)]
    runtime += [fingerprint(p) for p in sorted(exe.parent.glob("*.dll"))]
    out.mkdir()
    stages = {}
    for kind, rows in selected.items():
        inputs, questions = make_inputs(rows)
        input_path, question_path = out / f"{kind}.jsonl", out / f"{kind}.questions.jsonl"
        write_jsonl(input_path, inputs)
        write_jsonl(question_path, questions)
        token_path = out / f"{kind}.tokens.jsonl"
        with token_path.open("xb") as stdout, (out / f"{kind}.tokens.stderr.txt").open("xb") as stderr:
            subprocess.run([str(dump), str(Path(args.model).resolve()), "--batch", str(input_path)],
                           stdout=stdout, stderr=stderr, check=True)
        counts = check_token_counts(read_jsonl(token_path), inputs)
        stages[kind] = {"ids": [r["id"] for r in rows], "input": fingerprint(input_path),
                        "questions": fingerprint(question_path), "tokens": fingerprint(token_path), "prompt_lengths": counts,
                        "paragraph_counts": dict(sorted(Counter(len(r["context"]["title"]) for r in rows).items()))}
    protocol = {
        "schema_version": 1, "kind": KIND, "profile": PROFILE,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"repo": "hotpotqa/hotpot_qa", "revision": DATA_REV, "config": "distractor", "split": "validation",
                    "url": f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{DATA_REV}/distractor/{DATA_FILE}",
                    "rows": 7405, "source": fingerprint(raw / DATA_FILE), "license": "CC BY-SA 4.0"},
        "sampling": {"algorithm": "sort by SHA256(UTF8(f'hotpotqa-qwen08-v1:42:{id}')), tie by ID; first 200 pilot, next 500 main",
                     "seed": SEED, "consults_answers": False, "replacement_after_length_or_answer_checks": False},
        "context": {"paragraphs": "all supplied by the original distractor record", "nominal_paragraphs": 10,
                    "short_context_policy": "user approved keeping the same IDs and every available paragraph after the original-JSON audit; no padding or replacement",
                    "order": "original dataset order", "sentences": "concatenated unchanged",
                    "gold_document_selection": False, "retrieval": False, "prompt_prefix": PROMPT_PREFIX, "instruction": INSTRUCTION},
        "flags": flags, "generation": "stock non-thinking template, raw_argmax, no verification pass",
        "model": model, "binary": {"commit": active["commit"], "build": active["build"], "runtime": runtime},
        "source_probe": fingerprint(PROBE_PATH), "source_probe_meta": probe["info"],
        "source_config": fingerprint(PROBE_PATH.parent / "config.json"),
        "source_audit": fingerprint(SOURCE_AUDIT_PATH),
        "source_binary": {"commit": source_config["llama_cpp_commit"], "build": source_config["llama_cpp_build"]},
        "binary_caveat": "Source: 3ad2230/build 37; target: 785322d17/build 12033. Prior master control had 75.0% answer and 97.3% label agreement; this is not a bit-exact dataset-only transfer.",
        "evaluation": {"label_column": "hotpot_em", "required_splits": ["test"], "expected_rows": COUNTS,
                       "official_evaluator": {"revision": EVAL_REV, "url": f"https://raw.githubusercontent.com/hotpotqa/hotpot/{EVAL_REV}/hotpot_evaluate_v1.py",
                                              "source": fingerprint(raw / EVAL_FILE)},
                       "accuracy": "official answer EM and mean F1 on every attempt; whole raw answer, including truncated output; missing/empty output gets zero",
                       "primary": "status=ok, generated=true, not truncated, non-empty answer, valid C states and first-token scores; same rows for all methods",
                       "scores": ["frozen_probe_logit", "first_token_logprob", "negative_first_token_entropy"],
                       "higher_score": "more likely correct", "retraining": False, "calibration": False, "layer_selection": False,
                       "bootstrap": {"n": N_BOOT, "seed": SEED, "interval": "percentile 95%, resample questions; paired deltas"},
                       "selective_accuracy": {"retained_fraction": 0.5, "rounding": "ceil", "ties": "original input row order"}},
        "stages": stages,
        "implementation": [fingerprint(__file__), fingerprint(PIPELINE / "apply_probe_model.py"),
                           fingerprint(PIPELINE / "baseline.py"), fingerprint(PIPELINE / "verify_run.py"),
                           fingerprint(PIPELINE / "gate_report.py"), fingerprint(ROOT / "scripts/run.ps1"),
                           fingerprint(ROOT / active["dir"] / "examples/hidden-states/hidden-states.cpp"),
                           fingerprint(ROOT / active["dir"] / "examples/hidden-states/prompt_dump.cpp")],
    }
    write_json(out / "protocol.json", protocol)
    print(f"frozen protocol: {out / 'protocol.json'}")
    for kind, stage in stages.items():
        print(f"{kind}: {len(stage['ids'])} disjoint IDs, prompt tokens {stage['prompt_lengths']['min']}..{stage['prompt_lengths']['max']}")


def check_protocol(path, kind, input_path=None, questions_path=None, model_path=None):
    protocol = read_json(path)
    require(protocol.get("schema_version") == 1 and protocol.get("kind") == KIND and protocol.get("profile") == PROFILE,
            "not the frozen Qwen HotpotQA protocol")
    require(protocol["flags"] == profile_flags(), "profile flags changed since protocol freeze")
    require(protocol["source_probe"]["sha256"] == PROBE_SHA, "wrong source probe")
    require(protocol["evaluation"]["expected_rows"] == COUNTS, "sample sizes changed")
    all_ids = protocol["stages"]["pilot"]["ids"] + protocol["stages"]["main"]["ids"]
    require(len(all_ids) == len(set(all_ids)) == sum(COUNTS.values()), "pilot/evaluation IDs overlap")
    stage = protocol["stages"][kind]
    inputs = read_jsonl(check_fingerprint(stage["input"], input_path))
    questions = read_jsonl(check_fingerprint(stage["questions"], questions_path))
    require(len(inputs) == COUNTS[kind] and [r["id"] for r in inputs] == stage["ids"] == [r["id"] for r in questions],
            "input/gold/sample ID order mismatch")
    require(all(set(r) == {"id", "question", "split", "dataset"} and r["split"] == "test" and r["dataset"] == "hotpotqa" for r in inputs),
            "generation JSONL contains annotations, answers or a non-test split")
    check_token_counts(read_jsonl(check_fingerprint(stage["tokens"])), inputs)
    check_fingerprint(protocol["source_probe"])
    check_fingerprint(protocol["source_config"])
    check_fingerprint(protocol["source_audit"])
    check_fingerprint(protocol["evaluation"]["official_evaluator"]["source"])
    check_fingerprint(protocol["model"], model_path)
    for record in protocol["binary"]["runtime"]:
        check_fingerprint(record)
    for record in protocol["implementation"]:
        check_fingerprint(record)
    return protocol, inputs, questions


def check_run(directory, protocol_path, kind, questions_path=None):
    protocol, inputs, questions = check_protocol(protocol_path, kind, questions_path=questions_path)
    directory = Path(directory)
    runner = read_json(directory / "runner.json")
    evaluation = runner.get("evaluation") or {}
    require(runner["profile"] == PROFILE and runner["stage"] == kind and runner["flags"] == FLAGS,
            "runner profile/stage/flags mismatch")
    require(evaluation.get("kind") == KIND and evaluation.get("protocol_sha256", "").lower() == sha256(protocol_path),
            "runner did not use this frozen protocol")
    require(runner["input_sha256"].lower() == protocol["stages"][kind]["input"]["sha256"] and
            runner["questions_sha256"].lower() == protocol["stages"][kind]["questions"]["sha256"], "runner input/gold hash mismatch")
    require(Path(runner["input"]).resolve() == Path(protocol["stages"][kind]["input"]["path"]).resolve() and
            Path(runner["questions"]).resolve() == Path(protocol["stages"][kind]["questions"]["path"]).resolve(), "runner input/gold path mismatch")
    config = read_json(directory / "config.json")
    require(Path(config["batch"]).resolve() == Path(runner["input"]).resolve(), "config batch does not match the frozen input")
    require(config["model_name"] == "Qwen3.5-0.8B" and Path(config["model"]).resolve() == Path(protocol["model"]["path"]).resolve(), "wrong target model")
    require(config["layers"] == [13] and config["n_layers"] == 24 and config["n_embd"] == 1024 and config["n_layers_selected"] == 1,
            "target is not the frozen C/model-layer-13 feature")
    require(config["generate"] is True and config["template"] is True and config["thinking"] is False and
            config["max_len"] == 8192 and config["max_new_tokens"] == 64 and config.get("sampling", {}).get("type") == "raw_argmax",
            "generation differs from the frozen raw-argmax protocol")
    require(config["llama_cpp_commit"] == protocol["binary"]["commit"] and config["llama_cpp_build"] == protocol["binary"]["build"],
            "target binary commit/build mismatch")
    meta = read_jsonl(directory / "meta.jsonl")
    require(config["n_examples"] == len(meta) == COUNTS[kind] and [r["id"] for r in meta] == [r["id"] for r in inputs], "incomplete or reordered run")
    require([r["row"] for r in meta] == list(range(len(meta))) and
            all(r["split"] == "test" and r.get("dataset") == "hotpotqa" for r in meta), "row/split/dataset mismatch")
    lengths = read_jsonl(protocol["stages"][kind]["tokens"]["path"])
    # C++ omits question/n_prompt: bind text by input hash, lengths by point A.
    require(all(r.get("status") != "ok" or
                (isinstance(r.get("pos_a"), int) and r["pos_a"] + 1 == tok["n_prompt"])
                for r, tok in zip(meta, lengths)), "actual prompt tokens differ from diagnostic")
    require(all(r.get("status") != "ok" or
                (isinstance(r.get("n_tokens"), int) and isinstance(r.get("pos_c"), int) and
                 r["pos_a"] <= r["pos_c"] == r["n_tokens"] - 1 and r["n_tokens"] <= FLAGS["max_len"])
                for r in meta), "inconsistent state token positions")
    return protocol, config, meta, questions


def completion_reason(rec):
    if rec.get("status") != "ok":
        return "state_skipped"
    if rec.get("generated") is not True:
        return "not_generated"
    if rec.get("truncated") is True:
        return "truncated"
    if not isinstance(rec.get("answer"), str) or not rec["answer"].strip():
        return "empty_answer"
    return "completed"


def label_rows(meta, questions):
    require([r["id"] for r in meta] == [r["id"] for r in questions], "meta/gold IDs differ")
    labels = []
    for rec, gold in zip(meta, questions):
        answer = rec.get("answer", "") if rec.get("generated") is True else ""
        if not isinstance(answer, str):
            answer = ""
        f1, precision, recall = f1_score(answer, gold["answer"])
        labels.append({"row": rec["row"], "id": rec["id"], "split": rec["split"],
                       "hotpot_em": int(exact_match_score(answer, gold["answer"])),
                       "hotpot_f1": f1, "hotpot_precision": precision, "hotpot_recall": recall,
                       "completed": int(completion_reason(rec) == "completed"), "completion_reason": completion_reason(rec)})
    return labels


def label(args):
    out = Path(args.dir) / "labels.csv"
    assert_absent(out)
    _, _, meta, questions = check_run(args.dir, args.protocol, args.kind, args.questions)
    labels = label_rows(meta, questions)
    write_csv(out, labels)
    print(f"official answer labels: {len(labels)} attempts, EM={np.mean([r['hotpot_em'] for r in labels]):.4f}, F1={np.mean([r['hotpot_f1'] for r in labels]):.4f}")


def mean_ci(values, n_boot=N_BOOT, seed=SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    return ci([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)])


def retained_accuracy(y, score):
    n = math.ceil(len(y) * 0.5)
    kept = np.argsort(-score, kind="mergesort")[:n]
    return {"retained": n, "correct": int(y[kept].sum()), "accuracy": float(y[kept].mean()), "coverage": n / len(y)}


def paired_delta(y, probe_score, baseline_score, n_boot=N_BOOT, seed=SEED):
    out = {name: {"delta": fn(y, probe_score) - fn(y, baseline_score)} for name, fn in (("auroc", auroc), ("auprc", auprc))}
    draws = {name: [] for name in out}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() in (0, len(yb)):
            continue
        for name, fn in (("auroc", auroc), ("auprc", auprc)):
            draws[name].append(fn(yb, probe_score[idx]) - fn(yb, baseline_score[idx]))
    for name in out:
        out[name]["ci"] = ci(draws[name])
    return out


def primary_indices(meta, hidden, layer_index):
    """One answer-independent validity mask for the probe and both baselines."""
    reasons = []
    for rec in meta:
        reason = completion_reason(rec)
        if reason == "completed":
            state = np.asarray(hidden[rec["row"], layer_index], dtype=float)
            lp, entropy = rec.get("logprobs") or [], rec.get("entropies") or []
            if not np.isfinite(state).all() or not np.any(state):
                reason = "invalid_c_state"
            elif not lp or not entropy:
                reason = "missing_first_token_scores"
            elif not np.isfinite([lp[0], entropy[0]]).all():
                reason = "invalid_first_token_scores"
            else:
                reason = "included"
        reasons.append(reason)
    idx = np.asarray([r["row"] for r, reason in zip(meta, reasons) if reason == "included"], dtype=np.int64)
    return idx, reasons


def score(args):
    directory = Path(args.dir)
    out, scores_out = Path(args.out), Path(args.scores_out)
    assert_absent(out, scores_out)
    protocol, config, meta, questions = check_run(directory, args.protocol, "main")
    require(read_json(directory / "verify.json")["fails"] == 0, "verify failed; transfer scoring is blocked")
    require(read_json(directory / "gate.json")["passed"] is True, "run gate failed; transfer scoring is blocked")
    with (directory / "labels.csv").open(encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    expected = label_rows(meta, questions)
    require(len(labels) == len(expected) and all(
        str(r["row"]) == saved["row"] and r["id"] == saved["id"] and saved["split"] == "test" and
        str(r["hotpot_em"]) == saved["hotpot_em"] and float(saved["hotpot_f1"]) == r["hotpot_f1"] and
        str(r["completed"]) == saved["completed"] for r, saved in zip(expected, labels)), "saved official labels are inconsistent")
    probe = load_probe_model(protocol["source_probe"]["path"])
    hidden = np.load(directory / "hidden_c.npy", mmap_mode="r", allow_pickle=False)
    require(hidden.shape == (COUNTS["main"], 1, 1024) and hidden.dtype == np.float16, "wrong target hidden shape/dtype")
    layer_index = config["layers"].index(probe["info"]["layer_id"])
    idx, analysis_reasons = primary_indices(meta, hidden, layer_index)
    y = np.asarray([expected[i]["hotpot_em"] for i in idx], dtype=float)
    require(len(y) > 0 and 0 < y.sum() < len(y), "primary completed cohort must contain both classes")
    probe_score, layer_index = score_hidden(probe, hidden, idx, config["layers"])
    scores = {"probe": probe_score, "logprob": np.asarray([meta[i]["logprobs"][0] for i in idx]),
              "negative_entropy": -np.asarray([meta[i]["entropies"][0] for i in idx])}
    require(all(np.isfinite(s).all() for s in scores.values()), "non-finite confidence score")
    all_em = np.asarray([r["hotpot_em"] for r in expected], dtype=float)
    all_f1 = np.asarray([r["hotpot_f1"] for r in expected], dtype=float)
    results = {
        "kind": KIND, "protocol": fingerprint(args.protocol), "source_probe": protocol["source_probe"],
        "frozen_feature": {"point": "C", "layer_id": 13, "target_array_index": layer_index, "lambda": 1000.0,
                           "standardization": "unchanged saved train mu/sd", "fit_on_target": False, "calibration": False},
        "dataset": protocol["dataset"], "binary_caveat": protocol["binary_caveat"],
        "accuracy_all_attempts": {"n": len(meta), "correct": int(all_em.sum()), "em": float(all_em.mean()), "em_ci": mean_ci(all_em),
                                  "f1": float(all_f1.mean()), "f1_ci": mean_ci(all_f1)},
        "coverage": {"attempted": len(meta), "primary": len(idx), "primary_fraction": len(idx) / len(meta),
                     "completed": sum(r["completed"] for r in expected),
                     "completion_reasons": dict(Counter(completion_reason(r) for r in meta)),
                     "analysis_reasons": dict(Counter(analysis_reasons)),
                     "truncated": sum(r.get("truncated") is True for r in meta),
                     "empty_answers": sum(not isinstance(r.get("answer"), str) or not r["answer"].strip() for r in meta)},
        "primary": {"n": len(y), "correct": int(y.sum()), "em": float(y.mean()), "f1": float(all_f1[idx].mean()),
                    "label_column": "hotpot_em", "same_rows_for_all_methods": True},
        "metrics": {name: metrics(y, s, n_boot=N_BOOT, seed=SEED) for name, s in scores.items()},
        "paired_deltas": {f"probe_vs_{name}": paired_delta(y, probe_score, scores[name]) for name in ("logprob", "negative_entropy")},
        "retained_accuracy_50": {name: retained_accuracy(y, s) for name, s in scores.items()},
        "bootstrap": protocol["evaluation"]["bootstrap"],
        "artifacts": {name: fingerprint(directory / name) for name in ("runner.json", "config.json", "meta.jsonl", "labels.csv", "hidden_c.npy", "verify.json", "gate.json")},
        "implementation": fingerprint(__file__),
    }
    score_rows = [dict(r, analysis_included=int(reason == "included"), analysis_reason=reason,
                       frozen_probe_logit="", first_token_logprob="", negative_first_token_entropy="")
                  for r, reason in zip(expected, analysis_reasons)]
    for j, i in enumerate(idx):
        score_rows[i].update(frozen_probe_logit=format(probe_score[j], ".17g"), first_token_logprob=format(scores["logprob"][j], ".17g"),
                             negative_first_token_entropy=format(scores["negative_entropy"][j], ".17g"))
    require(sha256(PROBE_PATH) == PROBE_SHA, "source probe changed during scoring")
    write_csv(scores_out, score_rows)
    results["scores"] = fingerprint(scores_out)
    write_json(out, results)
    print(f"all {len(meta)} attempts: EM={all_em.mean():.4f}, F1={all_f1.mean():.4f}; primary={len(idx)}/{len(meta)}")
    for name, result in results["metrics"].items():
        print(f"{name}: AUROC={result['auroc']:.4f} {result.get('auroc_ci')}, AUPRC={result['auprc']:.4f} {result.get('auprc_ci')}")
    print(f"written: {out}, {scores_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="freeze disjoint inputs and diagnose every prompt without loading weights")
    prep.add_argument("--model", required=True)
    prep.add_argument("--raw-dir", default=str(PIPELINE / "data/hotpotqa_raw"))
    prep.add_argument("--out-dir", default=str(PIPELINE / "data/hotpotqa_qwen08_v2"))
    for name in ("check", "label", "score"):
        parser = commands.add_parser(name)
        parser.add_argument("--protocol", required=True)
        if name != "score":
            parser.add_argument("--kind", choices=("pilot", "main"), required=True)
            parser.add_argument("--questions")
        if name == "check":
            parser.add_argument("--input")
            parser.add_argument("--model")
        else:
            parser.add_argument("--dir", required=True)
        if name == "score":
            parser.add_argument("--out", required=True)
            parser.add_argument("--scores-out", required=True)
    args = ap.parse_args()
    try:
        if args.command == "check":
            check_protocol(args.protocol, args.kind, args.input, args.questions, args.model)
            print(f"frozen {args.kind} protocol/input/model/probe/runtime checks passed")
        else:
            {"prepare": prepare, "label": label, "score": score}[args.command](args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        ap.exit(1, f"HotpotQA: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
