"""Pilot/main gate for a llama-hidden-states run directory.

The runner calls this after every generation pass. It turns the checks that used to
live only in prose (all rows processed, no skipped rows, truncation share, empty
answers, channel/meta-text leak, array shape, throughput estimate) into one exit code.

A run is only allowed to continue to 6000 rows when `pilot` returns 0.

Usage:
    python gate_report.py --dir out/pilot_x --input data/pilot.jsonl \
        --log out/pilot_x.log --profile standard --stage pilot \
        --json out/pilot_x/gate.json

Exit code 0 if no hard check failed, 1 otherwise.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

DEFAULT_THRESHOLDS = {
    "max_skipped": 0,
    "max_truncated_ratio": 0.10,
    "max_empty_answer_ratio": 0.02,
    "max_channel_leak": 0,
    "max_repetition_ratio": 0.10,
    "max_long_answer_ratio": 0.15,
    "max_retokenize_ratio": 0.05,
    "max_estimated_hours": 6.0,
}

# Tokens that mean the model is still emitting protocol text instead of an answer.
# Both pipe spellings are listed: the broken Gemma 4 12B run printed '<|channel>thought'
# and '<channel|>' while the healthy gpt-oss run prints '<|channel|>'.
CHANNEL_MARKERS = [
    "<|channel|>",
    "<|channel>",
    "<channel|>",
    "<|start|>",
    "<|message|>",
    "<|end|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<start_of_turn>",
    "<end_of_turn>",
]
# Weaker signal: a thinking block leaked into the answer.
THOUGHT_MARKERS = ["<|channel|>thought", "<think>", "</think>", "thought\n", "\nthought"]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path, "rb") as f:
        for i, raw_line in enumerate(f, 1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SystemExit(f"{path}:{i}: invalid UTF-8: {exc}") from exc
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON: {e}")
    return rows


def read_labels(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_text_auto(path):
    """Read new UTF-8 logs and historical PowerShell UTF-16 logs."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def npy_shape(path):
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        return list(np.load(path, mmap_mode="r").shape)
    except Exception:
        return "unreadable"


def parse_log(path):
    text = ""
    if path and os.path.exists(path):
        text = read_text_auto(path)

    def num(pattern, group=1):
        m = re.search(pattern, text)
        return None if m is None else m.group(group)

    out = {
        "found": bool(text),
        "done": None,
        "retokenize_mismatch": None,
        "retokenize_total": None,
        "ex_per_s": None,
        "seconds": None,
        "layers_with_lout": None,
        "layers_requested": None,
        "arrow": None,
    }
    m = re.search(
        r"done: (\d+)/(\d+)\s+ok=(\d+) skipped=(\d+) truncated=(\d+)\s+in ([\d.]+) s \(([\d.]+) ex/s\)",
        text,
    )
    if m:
        out["done"] = {
            "rows": int(m.group(1)),
            "expected": int(m.group(2)),
            "ok": int(m.group(3)),
            "skipped": int(m.group(4)),
            "truncated": int(m.group(5)),
            "seconds": float(m.group(6)),
            "ex_per_s": float(m.group(7)),
        }
        out["seconds"] = float(m.group(6))
        out["ex_per_s"] = float(m.group(7))
    m = re.search(r"retokenize mismatch: (\d+)/(\d+)", text)
    if m:
        out["retokenize_mismatch"] = int(m.group(1))
        out["retokenize_total"] = int(m.group(2))
    m = re.search(r"selected layers after l_out probe: (\d+) of (\d+) requested", text)
    if m:
        out["layers_with_lout"] = int(m.group(1))
        out["layers_requested"] = int(m.group(2))
    m = re.search(r"l_out is available on (\d+) layer", text)
    if m and out["layers_with_lout"] is None:
        out["layers_with_lout"] = int(m.group(1))
    for block in ("A", "C"):
        m = re.search(r"wrote .*hidden_%s\.npy\s+shape=\[(\d+), (\d+), (\d+)\]" % block.lower(), text)
        if m:
            out["arrow"] = [int(x) for x in m.groups()]
    return out


class Gate:
    def __init__(self, stage):
        self.stage = stage
        self.checks = []
        self.fails = 0
        self.warns = 0

    def add(self, name, ok, detail="", severity="fail", value=None):
        sev = severity
        if not ok and severity == "warn":
            self.warns += 1
        elif not ok:
            self.fails += 1
        entry = {"check": name, "ok": bool(ok), "severity": sev, "detail": detail}
        if value is not None:
            entry["value"] = value
        self.checks.append(entry)
        mark = "ok" if ok else ("WARN" if severity == "warn" else "FAIL")
        print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
        return ok

    def ratio(self, name, value, limit, detail, severity="fail"):
        return self.add(name, value <= limit, f"{detail} (limit {limit:g})", severity, value)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--input", default=None, help="batch JSONL used for this run")
    ap.add_argument("--log", default=None, help="runner log with the binary summary lines")
    ap.add_argument("--stage", choices=("pilot", "main"), default="pilot")
    ap.add_argument("--profile", default=None, help="profile id from profiles.json")
    ap.add_argument("--profiles-file", default=os.path.join(root, "profiles.json"))
    ap.add_argument("--label-column", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    profile = None
    if args.profile:
        profiles = read_json(args.profiles_file)
        profile = profiles.get("profiles", {}).get(args.profile)
        if profile is None:
            raise SystemExit(f"unknown profile '{args.profile}' in {args.profiles_file}")
        thresholds.update(profiles.get("defaults", {}).get("pilot_gate", {}))
        thresholds.update(profile.get("pilot_gate", {}))

    label_column = args.label_column or (profile or {}).get("evaluation", {}).get("label_column", "correct_filtered_b")
    main_rows = (profile or {}).get("evaluation", {}).get("expected_rows", {}).get("main", 6000)

    d = args.dir
    if not os.path.isdir(d):
        raise SystemExit(f"no such directory: {d}")

    cfg_path = os.path.join(d, "config.json")
    meta_path = os.path.join(d, "meta.jsonl")
    for p in (cfg_path, meta_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    cfg = read_json(cfg_path)
    meta = read_jsonl(meta_path)
    labels = read_labels(os.path.join(d, "labels.csv"))
    log = parse_log(args.log)
    g = Gate(args.stage)

    # ----- structure -----
    n_cfg = int(cfg.get("n_examples") or 0)
    g.add("meta rows == config n_examples", len(meta) == n_cfg, f"{len(meta)} vs {n_cfg}")

    if args.input:
        if not os.path.exists(args.input):
            raise SystemExit(f"no such input file: {args.input}")
        expected = read_jsonl(args.input)
        g.add("meta rows == input rows", len(meta) == len(expected), f"{len(meta)} vs {len(expected)}")
        input_ids = [str(r.get("id")) for r in expected]
        meta_ids = [str(r.get("id")) for r in meta]
        g.add("meta id order matches input", meta_ids == input_ids,
              "first mismatch: " + str(next((i for i, pair in enumerate(zip(meta_ids, input_ids))
                                             if pair[0] != pair[1]), None)))
        if all("split" in r for r in expected):
            input_splits = [r.get("split") for r in expected]
            meta_splits = [r.get("split") for r in meta]
            g.add("meta splits match input", meta_splits == input_splits,
                  f"input {Counter(input_splits)} vs meta {Counter(meta_splits)}")
        dup = [k for k, v in Counter(str(r.get("id")) for r in expected).items() if v > 1]
        g.add(
            "input duplicate ids are the known ones",
            dup in ([], ["bb_956"]),
            f"duplicates: {dup}" if dup else "none",
            severity="warn",
        )

    rows = [int(r.get("row", -1)) for r in meta]
    contiguous = rows == list(range(len(rows)))
    g.add("row indices contiguous from 0", contiguous, f"{len(rows)} rows")

    status = Counter(str(r.get("status", "")) for r in meta)
    skipped = status.get("skipped", 0) + status.get("error", 0)
    ok = status.get("ok", 0)
    g.add("skipped rows", skipped <= thresholds["max_skipped"], f"{skipped} skipped")
    g.add("all rows ok", ok == len(meta), f"{ok} ok of {len(meta)}")

    truncated = sum(1 for r in meta if r.get("truncated") is True)
    trunc_ratio = truncated / len(meta) if meta else 1.0
    g.ratio(
        "truncated share",
        trunc_ratio,
        thresholds["max_truncated_ratio"],
        f"{truncated}/{len(meta)}",
        "fail",
    )

    generated = [r for r in meta if r.get("generated") is True]
    if cfg.get("generate") is True:
        g.add("all rows were generated", len(generated) == len(meta),
              f"{len(generated)}/{len(meta)}")
    empty = [r for r in generated if not str(r.get("answer", "")).strip()]
    empty_ratio = len(empty) / len(generated) if generated else 0.0
    g.ratio(
        "empty answers",
        empty_ratio,
        thresholds["max_empty_answer_ratio"],
        f"{len(empty)}/{len(generated)} generated",
        "fail",
    )

    leaks = []
    thoughts = []
    for r in meta:
        ans = str(r.get("answer", ""))
        if any(m in ans for m in CHANNEL_MARKERS):
            leaks.append(r.get("id"))
        if any(m in ans.lower() for m in THOUGHT_MARKERS):
            thoughts.append(r.get("id"))
    g.add(
        "no chat-protocol leak in answers",
        len(leaks) <= thresholds["max_channel_leak"],
        f"{len(leaks)} rows" + (f", e.g. {leaks[:3]}" if leaks else ""),
    )
    g.add(
        "no thinking block in answers",
        len(thoughts) == 0,
        f"{len(thoughts)} rows" + (f", e.g. {thoughts[:3]}" if thoughts else ""),
        severity="warn",
    )

    answers = [str(r.get("answer", "")).strip() for r in generated if str(r.get("answer", "")).strip()]
    if len(answers) >= 20:
        top, count = Counter(answers).most_common(1)[0]
        share = count / len(answers)
        g.add(
            "answers are not degenerate",
            share < 0.90,
            f"most common answer {share:.0%} of rows: {top[:40]!r}",
            severity="warn",
            value=round(share, 4),
        )

        # A broken prompt shows up as looping text: the 12B run repeats one word in
        # 28.8% of its long answers, healthy runs stay under 4%.
        repetitive = 0
        considered = 0
        for ans in answers:
            words = re.findall(r"\w+", ans.lower())
            if len(words) < 10:
                continue
            considered += 1
            if Counter(words).most_common(1)[0][1] / len(words) > 0.35:
                repetitive += 1
        rep_ratio = repetitive / considered if considered else 0.0
        g.ratio(
            "answers are not looping",
            rep_ratio,
            thresholds["max_repetition_ratio"],
            f"{repetitive}/{considered} long answers repeat one word",
            "fail",
        )

        long_ratio = sum(1 for ans in answers if len(ans) > 100) / len(answers)
        g.ratio(
            "share of very long answers",
            long_ratio,
            thresholds["max_long_answer_ratio"],
            f"{sum(1 for a in answers if len(a) > 100)}/{len(answers)} over 100 chars",
            "fail",
        )

    # ----- arrays -----
    shapes = {}
    for point in ("a", "c"):
        path = os.path.join(d, f"hidden_{point}.npy")
        if not os.path.exists(path):
            g.add(f"file hidden_{point}.npy", os.path.exists(path), path)
            continue
        shape = npy_shape(path)
        shapes[point] = shape
        if not isinstance(shape, list) or len(shape) != 3:
            g.add(f"hidden_{point}.npy header", False, str(shape))
            continue
        want = [len(meta), int(cfg.get("n_layers_selected") or 0), int(cfg.get("n_embd") or 0)]
        g.add(f"hidden_{point}.npy shape", shape == want, f"{shape} vs {want}")
    if "a" in shapes and "c" in shapes and isinstance(shapes["a"], list) and isinstance(shapes["c"], list):
        g.add("hidden_a and hidden_c shapes agree", shapes["a"] == shapes["c"], f"{shapes['a']} vs {shapes['c']}")

    requested = cfg.get("layers_requested")
    if requested:
        g.add(
            "selected layers == layers with l_out",
            int(cfg.get("n_layers_selected") or 0) == len(cfg.get("layers") or []),
            f"{len(cfg.get('layers') or [])} of {len(requested)} requested",
        )
        if profile and profile.get("expected", {}).get("layers_with_lout"):
            want = int(profile["expected"]["layers_with_lout"])
            g.add(
                "l_out layers match the profile",
                int(cfg.get("n_layers_selected") or 0) == want,
                f"{cfg.get('n_layers_selected')} vs expected {want}",
            )

    if log["layers_with_lout"] is not None and log["layers_requested"] is not None:
        g.add(
            "log: l_out probe",
            log["layers_with_lout"] == int(cfg.get("n_layers_selected") or 0),
            f"{log['layers_with_lout']} of {log['layers_requested']} requested",
            severity="warn",
        )

    # ----- throughput / log -----
    # The arrays/meta above are authoritative, but the final binary summary catches
    # a truncated or wrong log and guards against parsing the wrong run's log file.
    if log["done"] is None:
        g.add("log has the done summary", False, f"no 'done:' line in {args.log}", severity="warn")
    else:
        done = log["done"]
        g.add("log: processed rows match meta", done["rows"] == len(meta),
              f"{done['rows']} vs {len(meta)}")
        g.add("log: expected rows match config", done["expected"] == n_cfg,
              f"{done['expected']} vs {n_cfg}")
        g.add("log: ok/skipped match meta",
              done["ok"] == ok and done["skipped"] == skipped,
              f"log {done['ok']}/{done['skipped']} vs meta {ok}/{skipped}")
        g.add("log: truncated matches meta", done["truncated"] == truncated,
              f"{done['truncated']} vs {truncated}")

    if log["arrow"] is not None and isinstance(shapes.get("a"), list):
        g.add("log: written shape matches arrays", log["arrow"] == shapes["a"],
              f"{log['arrow']} vs {shapes['a']}", severity="warn")

    if log["retokenize_mismatch"] is None:
        g.add("log has retokenize summary", False,
              f"no 'retokenize mismatch:' line in {args.log}", severity="warn")
    elif log["retokenize_total"]:
        ratio = log["retokenize_mismatch"] / log["retokenize_total"]
        g.ratio(
            "retokenize mismatch share",
            ratio,
            thresholds["max_retokenize_ratio"],
            f"{log['retokenize_mismatch']}/{log['retokenize_total']}",
            severity="warn",
        )
    else:
        g.add("retokenize mismatch summary", True,
              f"{log['retokenize_mismatch']}/0 (no retokenized rows)")

    if log["ex_per_s"]:
        hours = float(main_rows) / log["ex_per_s"] / 3600.0
        if args.stage == "pilot":
            g.add(
                "estimated full run is affordable",
                hours <= thresholds["max_estimated_hours"],
                f"{hours:.2f} h for {main_rows} rows at {log['ex_per_s']:.2f} ex/s (limit {thresholds['max_estimated_hours']:g} h)",
                severity="fail",
                value=round(hours, 3),
            )
        else:
            g.add("throughput recorded", True, f"{log['ex_per_s']:.2f} ex/s")

    # ----- labels (optional, present after the label stage) -----
    if labels is not None:
        g.add("correctness label column exists", all(label_column in r for r in labels), label_column)
        g.add("correctness labels are binary or empty", all(r.get(label_column, "") in ("", "0", "1") for r in labels))
        nulls = sum(1 for r in labels if not str(r.get(label_column, "")).strip())
        pos = sum(1 for r in labels if str(r.get(label_column, "")).strip() == "1")
        labeled = len(labels) - nulls
        g.add("labels rows == meta rows", len(labels) == len(meta), f"{len(labels)} vs {len(meta)}")
        g.add(
            "both label classes present",
            pos > 0 and labeled - pos > 0,
            f"{pos} positive, {labeled - pos} negative, {nulls} null",
            severity="fail" if args.stage == "pilot" else "warn",
        )
        g.add(
            "null labels are explained by truncated rows" if label_column == "correct_filtered_b" else "official labels cover all attempted rows",
            nulls == (truncated if label_column == "correct_filtered_b" else 0),
            f"{nulls} null vs {truncated} truncated; column {label_column}",
            severity="warn" if label_column == "correct_filtered_b" else "fail",
        )

    summary = {
        "rows": len(meta),
        "ok": ok,
        "skipped": skipped,
        "truncated": truncated,
        "truncated_ratio": round(trunc_ratio, 4),
        "empty_answers": len(empty),
        "channel_leaks": len(leaks),
        "thought_leaks": len(thoughts),
        "ex_per_s": log["ex_per_s"],
        "seconds": log["seconds"],
        "estimated_hours_6000": round(6000.0 / log["ex_per_s"] / 3600.0, 3) if log["ex_per_s"] else None,
        "shape": shapes.get("a"),
    }
    if main_rows != 6000:
        summary["main_rows"] = main_rows
        summary["estimated_hours_main"] = round(float(main_rows) / log["ex_per_s"] / 3600.0, 3) if log["ex_per_s"] else None
    report = {
        "dir": d,
        "stage": args.stage,
        "profile": args.profile,
        "label_column": label_column,
        "input": args.input,
        "log": args.log,
        "passed": g.fails == 0,
        "fails": g.fails,
        "warns": g.warns,
        "thresholds": thresholds,
        "summary": summary,
        "checks": g.checks,
    }

    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = args.json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, args.json)

    verdict = "PASSED" if report["passed"] else "FAILED"
    print(f"\ngate {args.stage} {verdict}: {g.fails} failed, {g.warns} warnings")
    if summary["estimated_hours_6000"] is not None:
        print(f"full-run estimate: {summary.get('estimated_hours_main', summary['estimated_hours_6000'])} h for {main_rows} rows")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
