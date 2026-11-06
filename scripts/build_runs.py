"""Build RUNS.json from the tracked run artifacts and a small curated catalog.

Numbers are never copied from README/report: labels, shapes, baseline and probe metrics
come from pipeline/out. Only facts that cannot be inferred from files are curated here:
whether a run is valid/canonical, its human model name, its prompt regime and the note
explaining why it exists.

Usage:
    python scripts/build_runs.py           # atomically rewrite RUNS.json
    python scripts/build_runs.py --check   # fail if RUNS.json is stale
    python scripts/build_runs.py --stdout  # print the generated document
"""

import argparse
import csv
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "pipeline", "out")
DEFAULT_OUTPUT = os.path.join(ROOT, "RUNS.json")


# status/canonical/prompt interpretation is scientific provenance, not an artifact
# metric. Keep it explicit and review changes to this table like changes to a protocol.
MAIN = {
    "main_qwen08": dict(
        family="qwen3.5", model="Qwen3.5-0.8B BF16", status="valid", canonical=False,
        prompt_regime="default_bos", usable=True,
        notes="BF16 anchor of the quantization check. The size series uses Q4_K_M; keep this run for the BF16-vs-Q4 comparison only.",
    ),
    "main_qwen08_q4": dict(
        family="qwen3.5", model="Qwen3.5-0.8B Q4_K_M", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Anchor of the Qwen 3.5 size series. Binary 3ad2230 predates the BOS fix but add_bos_token is false for Qwen, so the prompt is unaffected.",
    ),
    "main_qwen08_q4_master": dict(
        family="qwen3.5", model="Qwen3.5-0.8B Q4_K_M", status="valid", canonical=False,
        prompt_regime="default_bos", usable=True,
        notes="Master-binary control of main_qwen08_q4. Paired dAUROC -0.0054 [-0.0225, +0.0114]; answers/labels agree 75.0%/97.3%. Use for binary-version checks, not as the series point.",
    ),
    "main_qwen35_2b_q4": dict(
        family="qwen3.5", model="Qwen3.5-2B Q4_K_M", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="One degenerate answer on dpql_1540 (row 2083, answer '#', empty label). Row left in place.",
    ),
    "main_qwen35_4b_q4": dict(
        family="qwen3.5", model="Qwen3.5-4B Q4_K_M", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Plateau partner of 9B: 0.9180 vs 0.9172, difference covers zero.",
    ),
    "main_qwen35_9b_q4": dict(
        family="qwen3.5", model="Qwen3.5-9B Q4_K_M", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Largest size of the series that fits 6 GB VRAM. 0 truncated, 0 empty answers.",
    ),
    "main_gemma4e2b": dict(
        family="gemma4-small", model="Gemma 4 E2B Q4_K_M", status="valid", canonical=True,
        prompt_regime="legacy_no_bos", usable=True,
        notes="Only E2B run. Measured before the 2026-11-02 BOS fix, so the prompt lacks the leading <bos> and differs by one token from llama-cli. Answers, labels and probe passed controls; do not compare token-exact with BOS runs.",
    ),
    "main_gemma4e4b": dict(
        family="gemma4-small", model="Gemma 4 E4B Q4_K_M", status="valid", canonical=False,
        prompt_regime="legacy_no_bos", usable=True, superseded_by="main_gemma4e4b_bos_master",
        notes="Old no-BOS E4B run, kept for the prompt comparison: test 0.9312 vs 0.9186 for the BOS run, paired delta -0.0126 [-0.0259, +0.0012].",
    ),
    "main_gemma4e4b_bos_master": dict(
        family="gemma4-small", model="Gemma 4 E4B Q4_K_M", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Current E4B run: master binary, leading <bos> present, verify 0 fails. Replaces main_gemma4e4b for new comparisons.",
    ),
    "main_gemma4_12b": dict(
        family="gemma4-12b", model="Gemma-4-12B-It Q4_K_M", status="invalid", canonical=False,
        prompt_regime="legacy_no_bos", usable=False, superseded_by="main_gemma4_12b_v3",
        notes="Catastrophically broken prompt: no leading <bos>, answers are meta-text. Only 3702/6000 rows labeled, 4 positive, 2298 truncated. Never use for analysis. Kept only as evidence for the BOS incident.",
    ),
    "main_gemma4_12b_v3": dict(
        family="gemma4-12b", model="Gemma-4-12B-It Q4_K_M", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Valid Kaggle run after the BOS fix: 6000/6000, 21 truncated, verify 0 fails. Probe computed locally on 2026-11-04 (lambda 3000, C/28): val 0.9162, test 0.9107 [0.8950, 0.9262]; probe2 reproduced it byte-for-byte.",
    ),
    "main_gemma4_26b_a4b_udq4km": dict(
        family="gemma4-26b-a4b", model="Gemma 4 26B-A4B UD-Q4_K_M", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Valid Kaggle v2 run on two T4 GPUs: 6000/6000, 32 truncated, shape [6000, 30, 2816], verify 0 fails and 0 warnings. Probe lambda 3000, C/16: val 0.9018, test 0.9077 [0.8893, 0.9235]; probe2 reproduced both CSVs byte-for-byte.",
    ),
    "main_gptoss20b": dict(
        family="gpt-oss", model="gpt-oss-20b MXFP4", status="partial", canonical=False,
        prompt_regime="no_template_gptoss_final", usable=False, superseded_by="main_gptoss20b_v2",
        notes="Stopped by hand at 325/6000 rows; no labels.csv. Never use.",
    ),
    "main_gptoss20b_dupinstr": dict(
        family="gpt-oss", model="gpt-oss-20b MXFP4", status="invalid", canonical=False,
        prompt_regime="no_template_gptoss_final", usable=False, superseded_by="main_gptoss20b_v2",
        notes="Input builder duplicated the instruction inside the harmony prompt. Numbers look plausible but the prompt is wrong; never use.",
    ),
    "main_gptoss20b_v2": dict(
        family="gpt-oss", model="gpt-oss-20b MXFP4", status="valid", canonical=True,
        prompt_regime="no_template_gptoss_final", usable=True,
        notes="Valid gpt-oss run: pre-filled final channel, --no-template, -ngl 11, --batch-rows 8. 27 truncated, 10733.5 s.",
    ),
    "main_nemotron3nano4b": dict(
        family="nemotron_h", model="NVIDIA-Nemotron-3-Nano-4B", status="invalid", canonical=False,
        prompt_regime="legacy_no_bos", usable=False, superseded_by="main_nemotron3nano4b_v2",
        notes="42 layers requested, only 17 emit l_out. The old binary marked all 5269 processed rows skipped and wrote zeros; 0 labeled rows. Never use.",
    ),
    "main_nemotron3nano4b_v2": dict(
        family="nemotron_h", model="NVIDIA-Nemotron-3-Nano-4B", status="valid", canonical=True,
        prompt_regime="legacy_no_bos", usable=True,
        notes="Valid run with the fixed binary: requested 42 layers auto-pruned to 17 l_out layers, shape [6000, 17, 3136]. verify warns=1 because of the expected pruning. Prompt lacks <bos> (pre-fix), same caveat as main_gemma4e2b.",
    ),
    "main_lfm25_1p2b_instruct_q4": dict(
        family="lfm25", model="LFM2.5-1.2B Instruct Q4", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Instruct counterpart of the deferred Thinking experiment. 1 truncated row.",
    ),
    "main_lfm25_1p2b_thinking_r32_q4_v2": dict(
        family="lfm25", model="LFM2.5-1.2B-Thinking Q4 (32 reasoning tokens)", status="valid", canonical=True,
        prompt_regime="no_template_lfm_reasoning_final", usable=True,
        restriction="A is post-reasoning (last token of the forced Final answer: cue), so A is not token-equivalent to the Instruct run; C is directly comparable. 39/6000 rows have no extracted final and no label.",
        notes="First complete Thinking run: 32 generated reasoning tokens, forced final cue, 64 final tokens. 6000/6000 states, 5961 labeled (39 empty finals), verify 0 fails and 0 warnings. Probe lambda 1000, C/9: val 0.8819, test 0.8813 [0.8553, 0.9051]; probe2 reproduced both CSVs byte-for-byte. Against main_lfm25_1p2b_instruct_q4 on common test rows: delta AUROC +0.0016 [-0.0278, +0.0316], delta AUPRC -0.0365 [-0.1019, +0.0282].",
    ),
    "main_lfm25_2p6b_q4_final": dict(
        family="lfm25", model="LFM2.5-2.6B Q4", status="valid", canonical=True,
        prompt_regime="no_template_lfm_final", usable=True,
        notes="Stock template opens thinking even with thinking=false, so the input was rebuilt with make_lfm_final_prompt.py and run with --no-template. 128 truncated, 138 null of 6000.",
    ),
    "main_lfm3_1p7b_qad_q4": dict(
        family="lfm3", model="LFM3-1.7B QAD Q4", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        restriction="41.8% truncated: every metric uses the 3491 completed rows only.",
        notes="Style not resolved: raw argmax, unstable thinking=false and the 64-token budget are all plausible causes. Do not launch another full run before a llama-cli sampler diagnostic.",
    ),
    "main_yandexgpt5lite8b_q4": dict(
        family="yandexgpt", model="YandexGPT 5 Lite 8B Instruct Q4", status="valid", canonical=True,
        prompt_regime="default_bos", usable=True,
        notes="Master binary, leading <bos>, arch llama. Statistically indistinguishable from Qwen 9B and Gemma E4B.",
    ),
}


PILOTS = {
    "pilot_qwen08": dict(status="valid", prompt_regime="default_bos",
                          notes="Old binary sanity pilot of the 0.8B BF16 model."),
    "pilot_qwen08_q4": dict(status="valid", prompt_regime="default_bos",
                             notes="Q4 anchor pilot; predicted the full run time within 3%."),
    "pilot_qwen08_iq2": dict(status="invalid", prompt_regime="default_bos",
                              notes="UD-IQ2_XXS: 196/200 truncated (answers run into thinking). This quant is excluded from the series."),
    "pilot_qwen35_2b_q4": dict(status="valid", prompt_regime="default_bos",
                                notes="2B pilot, 28.7 s; full run 866.0 s."),
    "pilot_qwen35_4b_q4": dict(status="valid", prompt_regime="default_bos",
                                notes="4B pilot, 44.2 s; full run 1483.5 s."),
    "pilot_qwen35_9b_q4": dict(status="valid", prompt_regime="default_bos",
                                notes="9B pilot, 74.5 s with file logging; -ngl 99 fits 5860 MiB."),
    "pilot_gemma4e2b": dict(status="valid", prompt_regime="legacy_no_bos", notes="E2B pilot."),
    "pilot_gemma4e4b": dict(status="valid", prompt_regime="legacy_no_bos",
                             notes="E4B pilot, 43.1 s; predicted the full run within 6%."),
    "pilot_gemma4e4b_bos_master": dict(status="valid", prompt_regime="default_bos",
                                        notes="E4B BOS pilot on the master binary."),
    "pilot_gemma4_12b": dict(status="invalid", prompt_regime="legacy_no_bos",
                              notes="40-row 12B pilot with the broken prompt. Kept as evidence."),
    "pilot_nemotron3nano4b": dict(status="invalid", prompt_regime="legacy_no_bos",
                                   notes="Old binary: all rows skipped, zero states."),
    "pilot_nemotron3nano4b_v2": dict(status="valid", prompt_regime="legacy_no_bos",
                                      probe_status="pipeline_check",
                                      notes="Valid 200-row pilot. The probe in this directory is a 120/40/40 pipeline check, not a result."),
    "pilot_lfm25_1p2b_instruct_q4": dict(status="valid", prompt_regime="default_bos",
                                          notes="Instruct pilot, 200/200 ok, 0 truncated."),
    "pilot_lfm25_1p2b_thinking_q4": dict(status="invalid", prompt_regime="default_bos",
                                           notes="Thinking model: 200/200 truncated. Blocked until a reasoning budget is fixed."),
    "pilot_lfm25_1p2b_thinking_r32_q4": dict(
        status="partial", prompt_regime="no_template_lfm_reasoning_final",
        notes="Valid state replay, but the runner stopped before labeling because Invoke-PilotGate captured stdout with the exit code. Kept as evidence; superseded by the v2 pilot.",
    ),
    "pilot_lfm25_1p2b_thinking_r32_q4_v2": dict(
        status="valid", prompt_regime="no_template_lfm_reasoning_final",
        notes="Fixed reasoning protocol pilot: 32 reasoning tokens, forced final cue, 64 final tokens, 200/200 extracted, both label classes present; gate passed with one expected replay-log warning.",
    ),
    "pilot_lfm25_2p6b_q4_final": dict(status="valid", prompt_regime="no_template_lfm_final",
                                       notes="Final-cue pilot; 8 truncated of 200."),
    "pilot_lfm3_1p7b_qad_q4": dict(status="valid", prompt_regime="default_bos",
                                    notes="LFM3 pilot: 79/200 truncated, already showing the 64-token problem."),
    "pilot_yandexgpt5lite8b_q4": dict(status="valid", prompt_regime="default_bos",
                                       notes="Yandex pilot, 1 truncated."),
}


# These directories are intentionally ignored by git. Row counts are provenance, not
# analysis metrics, so keep them with the notes; a clean clone cannot inspect them.
AD_HOC = {
    "equiv": (None, "Pin-vs-master comparison logs and JSON files. No meta.jsonl; not a run directory."),
    "lfm_trials_2p6": (7, "7-row LFM2.5-2.6B prompt trial, no template, 1 truncated."),
    "lfm_trials_think": (7, "7-row LFM2.5-1.2B-Thinking trial at 64 tokens: 7/7 truncated."),
    "lfm_trials_think_256": (7, "7-row LFM2.5-1.2B-Thinking trial at 256 tokens: 5/7 still truncated."),
    "pilot20_gemma4e4b_bos_master": (20, "20-row BOS smoke test of the E4B prompt on the master binary."),
    "pilot20_lfm25_1p2b_inst_q4": (20, "20-row smoke test of the LFM2.5-1.2B Instruct template, 0 truncated."),
    "pilot20_lfm25_1p2b_think_q4": (20, "20-row Thinking trial with the stock template: 20/20 truncated."),
    "pilot20_lfm25_1p2b_think_q4_prefill": (20, "20-row Thinking trial with a pre-filled reply: still 20/20 truncated."),
    "pilot20_lfm25_2p6b_q4": (20, "20-row trial with the stock template: 20/20 truncated (the template opens a think block)."),
    "pilot20_lfm25_2p6b_q4_prefill": (20, "20-row trial with a pre-filled final cue: 4 truncated. Led to make_lfm_final_prompt.py."),
    "pilot20_lfm25_2p6b_q4_answer": (20, "20-row trial with the answer pre-filled in the input: 1 truncated."),
    "pilot20_lfm3_1p7b_final": (20, "20-row LFM3 trial with a Final answer cue: 12 truncated, so the cue was rejected."),
    "pilot20_lfm3_1p7b_qad_q4": (20, "20-row LFM3 smoke test with the stock template: 9 truncated."),
    "pilot20_yandexgpt5lite8b_q4": (20, "20-row YandexGPT smoke test, 0 truncated."),
}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path, allow_malformed=False):
    rows = []
    with open(path, encoding="utf-8", errors="strict") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    if allow_malformed:
                        # An interrupted partial run can end in a torn JSON line. It
                        # remains on disk as evidence and still counts toward the
                        # registry's physical meta_rows, but contributes no metrics.
                        rows.append({"_malformed": True, "_line": line_no})
                        continue
                    raise SystemExit(f"{path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{line_no}: JSON row is not an object")
                rows.append(row)
    return rows


def read_labels(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def round4(value):
    return None if value is None else round(float(value), 4)


def artifacts(name, allow_malformed=False):
    directory = os.path.join(OUT, name)
    if not os.path.isdir(directory):
        raise SystemExit(f"registered run directory is missing: {directory}")
    cfg = read_json(os.path.join(directory, "config.json"))
    meta = read_jsonl(os.path.join(directory, "meta.jsonl"), allow_malformed=allow_malformed)
    labels = read_labels(os.path.join(directory, "labels.csv"))
    baseline_path = os.path.join(directory, "baseline.json")
    probe_path = os.path.join(directory, "probe.json")
    verify_path = os.path.join(directory, "verify.json")
    return {
        "config": cfg,
        "meta": meta,
        "labels": labels,
        "baseline": read_json(baseline_path) if os.path.exists(baseline_path) else None,
        "probe": read_json(probe_path) if os.path.exists(probe_path) else None,
        "verify": read_json(verify_path) if os.path.exists(verify_path) else None,
    }


def label_summary(labels, meta):
    if labels is None:
        return {"rows": None, "labeled": None, "null": None,
                "truncated": None,
                "base_rate_filtered_b": None}
    values = [str(row.get("correct_filtered_b", "")).strip() for row in labels]
    labeled = sum(value in ("0", "1") for value in values)
    positive = sum(value == "1" for value in values)
    return {
        "rows": len(labels),
        "labeled": labeled,
        "null": len(labels) - labeled,
        "truncated": sum(bool(r.get("truncated")) for r in meta),
        "base_rate_filtered_b": round4(positive / labeled) if labeled else None,
    }


def baseline_summary(data):
    if data is None:
        return None
    test = (data.get("splits") or {}).get("test") or {}
    return {
        "logprob": round4((test.get("logprob") or {}).get("auroc")),
        "entropy": round4((test.get("entropy") or {}).get("auroc")),
    }


def probe_summary(data):
    if data is None:
        return None
    selected = data.get("selected") or {}
    test = (data.get("metrics") or {}).get("test") or {}
    return {
        "lambda": float(data["selected_lambda"]),
        "point": selected.get("point"),
        "layer": int(selected["layer"]),
        "layer_id": int(selected.get("layer_id", selected["layer"])),
        "val_auroc": round4(selected.get("val_auroc")),
        "test_auroc": round4(test.get("auroc")),
        "test_auprc": round4(test.get("auprc")),
        "test_n": int(test["n"]),
        "test_pos": int(test["pos"]),
    }


def binary_summary(cfg):
    return {
        "commit": cfg.get("llama_cpp_commit"),
        "build": cfg.get("llama_cpp_build"),
    }


def shape(cfg, rows):
    layers = cfg.get("n_layers_selected")
    embd = cfg.get("n_embd")
    return None if layers is None or embd is None else [rows, int(layers), int(embd)]


def main_entry(name, curated):
    data = artifacts(name, allow_malformed=curated["status"] != "valid")
    cfg, meta = data["config"], data["meta"]
    probe = probe_summary(data["probe"]) if curated["status"] == "valid" else None
    if curated["status"] != "valid":
        probe_status = "not_citable"
    elif probe is not None:
        probe_status = curated.get("probe_status", "complete")
    else:
        probe_status = curated.get("probe_status", "not_run")
    verify = data["verify"]
    entry = {
        "kind": "main",
        "family": curated["family"],
        "model": curated["model"],
        "status": curated["status"],
        "canonical": curated["canonical"],
        "usable_for_analysis": curated["usable"],
        "prompt_regime": curated["prompt_regime"],
        "restriction": curated.get("restriction"),
    }
    if curated.get("superseded_by"):
        entry["superseded_by"] = curated["superseded_by"]
    entry.update({
        "meta_rows": len(meta),
        "binary": binary_summary(cfg),
        "shape": shape(cfg, len(meta)),
        "labels": label_summary(data["labels"], meta),
        "baseline_test_auroc": baseline_summary(data["baseline"]),
        "probe": probe,
        "probe_status": probe_status,
        "verify": ({"status": ("passed" if int(verify.get("fails", 0)) == 0 else "failed"),
                    "fails": int(verify.get("fails", 0)),
                    "warns": int(verify.get("warns", 0))}
                   if verify is not None else {"status": "not_run"}),
        "notes": curated["notes"],
    })
    return entry


def pilot_entry(name, curated):
    data = artifacts(name, allow_malformed=curated["status"] != "valid")
    cfg, meta = data["config"], data["meta"]
    labels = label_summary(data["labels"], meta)
    return {
        "kind": "pilot",
        "status": curated["status"],
        "citable": False,
        "prompt_regime": curated["prompt_regime"],
        "meta_rows": len(meta),
        "shape": shape(cfg, len(meta)),
        "binary": binary_summary(cfg),
        "truncated": sum(bool(row.get("truncated")) for row in meta),
        "labels": labels,
        "probe_status": curated.get("probe_status", "not_run"),
        "notes": curated["notes"],
    }


def build_document():
    return {
        "schema_version": 1,
        "updated": "2026-11-06",
        "generator": "scripts/build_runs.py",
        "generated_from": "pipeline/out/*/{config.json,meta.jsonl,labels.csv,baseline.json,probe.json,verify.json}",
        "read_this_first": "docs/START-HERE.md",
        "status_legend": {
            "valid": "The run is internally consistent and its numbers may be cited.",
            "invalid": "The run is broken (wrong prompt, wrong binary, zero states). Never cite it.",
            "partial": "The run stopped early and has no complete artifacts. Never cite it.",
        },
        "canonical_legend": "canonical=true is the run to cite for that model under the current binary and prompt regime. Non-canonical valid runs stay available for the specific comparisons named in notes.",
        "prompt_regime_legend": {
            "default_bos": "Stock chat template, leading <bos> added by the tokenizer when the model requires it (current behaviour).",
            "legacy_no_bos": "Stock chat template but the required leading <bos> was lost (pre-2026-11-02 affected models).",
            "no_template_gptoss_final": "--no-template with a pre-filled harmony final channel (make_gptoss_prompt.py).",
            "no_template_lfm_final": "--no-template with a Final answer: cue (make_lfm_final_prompt.py).",
            "no_template_lfm_reasoning_final": "--no-template replay after 32 generated reasoning tokens and a forced Final answer: cue (lfm_reasoning.py).",
        },
        "probe_status_legend": {
            "complete": "probe.json is the frozen-protocol result.",
            "deferred": "Deliberately not computed. Do not run probe.py without a new explicit request.",
            "not_run": "No probe in this directory.",
            "not_citable": "The run itself is invalid or partial, so its probe numbers must not be cited even though files exist.",
            "pipeline_check": "Probe exists only to check the pipeline on a pilot; not a research result.",
        },
        "probe_protocol": "docs/START-HERE.md#frozen-probe-protocol",
        "shape_legend": "shape = [rows in meta.jsonl, selected l_out layers, n_embd]",
        "verify_legend": "verify.status=passed means verify.json has fails=0. It checks internal consistency of the directory, NOT that the prompt was correct: main_gemma4_12b passes verify and is still invalid. not_run means verify_run.py was never run on this directory.",
        "main_runs": {name: main_entry(name, curated) for name, curated in MAIN.items()},
        "pilots": {name: pilot_entry(name, curated) for name, curated in PILOTS.items()},
        "ad_hoc": {name: {"citable": False, "rows": rows, "notes": note}
                   for name, (rows, note) in AD_HOC.items()},
        "ad_hoc_legend": "Directories in pipeline/out that look like runs but are prompt trials or comparison logs. They are kept on disk and never cited. Only main_runs may be referenced as research results.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--check", action="store_true", help="do not write; fail if output differs")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    document = build_document()
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        if not os.path.exists(args.output):
            print(f"RUNS registry missing: {args.output}")
            return 1
        current = read_json(args.output)
        if current != document:
            print("RUNS.json is stale. Regenerate with:")
            print("  python scripts/build_runs.py")
            return 1
        print(f"RUNS.json is current: {len(MAIN)} main, {len(PILOTS)} pilot, {len(AD_HOC)} ad-hoc entries")
        return 0

    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, args.output)
    print(f"wrote {args.output}: {len(MAIN)} main, {len(PILOTS)} pilot, {len(AD_HOC)} ad-hoc entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
