"""Preflight checks for a batch JSONL before spending GPU time.

The expensive failures in this project were input failures: a duplicated
instruction added by a prompt builder, an empty question, a wrong row count. This
script checks those in under a second.

Usage:
    python preflight.py --input data/main.jsonl --expect-rows 6000 --require-instruction
    python preflight.py --input data/pilot_gptoss_final.jsonl --expect-rows 200 --require-instruction

Exit code 0 if no hard check failed, 1 otherwise.
"""

import argparse
import json
import os
import sys
from collections import Counter

INSTRUCTION = "Answer with just the answer, no explanation."
KNOWN_DUPLICATE_IDS = {"bb_956"}


class Report:
    def __init__(self):
        self.checks = []
        self.fails = 0
        self.warns = 0

    def add(self, name, ok, detail="", severity="fail"):
        if not ok:
            if severity == "warn":
                self.warns += 1
            else:
                self.fails += 1
        self.checks.append({"check": name, "ok": bool(ok), "severity": severity, "detail": detail})
        mark = "ok" if ok else ("WARN" if severity == "warn" else "FAIL")
        print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
        return ok


def check_input(path, expect_rows=None, require_instruction=False, allow_duplicates=None):
    """Return a report dict for one JSONL file. Never writes anything."""
    allow_duplicates = set(KNOWN_DUPLICATE_IDS if allow_duplicates is None else allow_duplicates)
    rep = Report()
    rep.add("input exists", os.path.exists(path), path)
    if not os.path.exists(path):
        return _report(path, rep, None)

    parsed = []
    bad_lines = []
    with open(path, "rb") as f:
        for i, raw_line in enumerate(f, 1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                bad_lines.append(f"{i}: invalid UTF-8 byte sequence")
                continue
            if not line.strip():
                continue
            try:
                parsed.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                bad_lines.append(f"{i}: {e.msg}")
    rep.add("every line is valid JSON", not bad_lines, "; ".join(bad_lines[:3]))
    bad_objects = [i for i, value in parsed if not isinstance(value, dict)]
    rep.add("every JSON row is an object", not bad_objects,
            f"non-object lines: {bad_objects[:5]}" if bad_objects else "")
    rows = [value for _, value in parsed if isinstance(value, dict)]
    rep.add("input is not empty", len(parsed) > 0, f"{len(parsed)} rows")
    if not rows:
        return _report(path, rep, None)

    rep.add("every row has id and question",
            all(isinstance(r.get("question"), str)
                and r.get("id") is not None
                and str(r.get("id")).strip() != "" for r in rows))
    empty = [str(r.get("id")) for r in rows if not str(r.get("question", "")).strip()]
    rep.add("no empty questions", not empty, f"{len(empty)} empty" + (f", e.g. {empty[:3]}" if empty else ""))

    if expect_rows is not None:
        rep.add("row count", len(rows) == expect_rows, f"{len(rows)} vs expected {expect_rows}")

    instr = [(r.get("question") if isinstance(r.get("question"), str) else "").count(INSTRUCTION)
             for r in rows]
    doubled = [str(r.get("id")) for r, c in zip(rows, instr) if c > 1]
    missing = sum(1 for c in instr if c == 0)
    rep.add("instruction is never duplicated", not doubled,
            f"{len(doubled)} rows" + (f", e.g. {doubled[:3]}" if doubled else ""))
    rep.add("instruction is present", missing == 0, f"{missing} rows without it",
            severity="fail" if require_instruction else "warn")

    ids = Counter(str(r.get("id")) for r in rows)
    dups = {k: v for k, v in ids.items() if v > 1}
    # A known duplicate means exactly two occurrences, not "any number is fine".
    unexpected = {k: v for k, v in dups.items() if k not in allow_duplicates or v != 2}
    rep.add("duplicate ids are the known ones", not unexpected,
            f"duplicates: {sorted(dups)}" if dups else "none",
            severity="warn" if not unexpected else "fail")

    lengths = [len(str(r.get("question", ""))) for r in rows]
    rep.add("question length is sane", max(lengths) < 20000, f"max {max(lengths)} chars", severity="warn")
    extra_keys = sorted({k for r in rows for k in r} - {"id", "question", "answer", "answer_gold"})
    rep.add("extra keys are carried into meta.jsonl", True, ", ".join(extra_keys) if extra_keys else "none",
            severity="warn")

    return _report(path, rep, rows)


def _report(path, rep, rows):
    summary = None
    if rows:
        questions = [r.get("question") if isinstance(r.get("question"), str) else "" for r in rows]
        summary = {
            "rows": len(rows),
            "first_id": str(rows[0].get("id")),
            "first_question": questions[0][:300],
            "last_id": str(rows[-1].get("id")),
            "last_question": questions[-1][-300:],
            "instruction_min": min(q.count(INSTRUCTION) for q in questions),
            "instruction_max": max(q.count(INSTRUCTION) for q in questions),
        }
    return {
        "input": path,
        "passed": rep.fails == 0,
        "fails": rep.fails,
        "warns": rep.warns,
        "summary": summary,
        "checks": rep.checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--expect-rows", type=int, default=None)
    ap.add_argument("--require-instruction", action="store_true")
    ap.add_argument("--allow-duplicate-id", action="append", default=None,
                    help="id that may appear twice (default: bb_956)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    report = check_input(args.input, args.expect_rows, args.require_instruction, args.allow_duplicate_id)
    if report["summary"]:
        s = report["summary"]
        print(f"\nfirst: {s['first_question']!r}")
        print(f"last:  {s['last_question']!r}")
        print(f"instruction occurrences: min {s['instruction_min']}, max {s['instruction_max']}")
    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = args.json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, args.json)
    print(f"\npreflight {'PASSED' if report['passed'] else 'FAILED'}: {report['fails']} failed, {report['warns']} warnings")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
