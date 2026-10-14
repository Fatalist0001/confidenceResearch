"""Label correctness of llama-hidden-states outputs against gold answers.

Normal mode:  label.py --meta <out_dir>/meta.jsonl --out labels.csv
Audit mode:   label.py --meta <out_dir>/meta.jsonl --audit N --seed S --out audit.csv
Score mode:   label.py --audit-score audit.csv
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict

from normalize import normalize


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def contains_phrase(ans_tokens, gold_norm):
    """True if gold_norm appears in ans as a contiguous token sequence."""
    if not gold_norm:
        return False
    gt = gold_norm.split()
    gl = len(gt)
    if gl == 0 or gl > len(ans_tokens):
        return False
    for i in range(len(ans_tokens) - gl + 1):
        if ans_tokens[i:i + gl] == gt:
            return True
    return False


def label_row(row):
    """Return (correct, n_answer_tokens). correct in {0, 1, None}."""
    status = row.get("status")
    generated = as_bool(row.get("generated", False))
    truncated = as_bool(row.get("truncated", False))
    answer = row.get("answer") or ""
    gold = row.get("gold") or []

    pos_a = row.get("pos_a")
    pos_c = row.get("pos_c")
    n_answer_tokens = None
    if isinstance(pos_a, int) and isinstance(pos_c, int) and pos_c >= pos_a:
        n_answer_tokens = pos_c - pos_a

    if status != "ok" or not generated:
        return None, n_answer_tokens
    if truncated:
        return None, n_answer_tokens
    ans_norm = normalize(answer)
    if not ans_norm:
        return None, n_answer_tokens
    ans_tokens = ans_norm.split()
    for g in gold:
        if contains_phrase(ans_tokens, g):
            return 1, n_answer_tokens
    return 0, n_answer_tokens


def read_meta(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_labels(rows):
    """Return list of (row, correct, n_answer_tokens)."""
    return [(r, *label_row(r)) for r in rows]


def write_labels_csv(labeled, out_path):
    fields = ["id", "row", "split", "correct", "truncated", "n_answer_tokens", "answer", "gold_main"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, correct, nat in labeled:
            w.writerow({
                "id": r.get("id", ""),
                "row": r.get("row", ""),
                "split": r.get("split", ""),
                "correct": "" if correct is None else correct,
                "truncated": r.get("truncated", ""),
                "n_answer_tokens": "" if nat is None else nat,
                "answer": r.get("answer", ""),
                "gold_main": r.get("gold_main", ""),
            })


def report(labeled):
    by_split = defaultdict(int)
    corr = defaultdict(lambda: [0, 0])
    total_correct = 0
    total_labeled = 0
    total_null = 0
    for r, correct, _nat in labeled:
        sp = r.get("split", "?")
        by_split[sp] += 1
        if correct is None:
            total_null += 1
        else:
            total_labeled += 1
            total_correct += correct
            corr[sp][0] += correct
            corr[sp][1] += 1

    print("=== rows by split ===")
    for sp in sorted(by_split):
        print(f"  {sp}: {by_split[sp]}")
    print(f"  total: {sum(by_split.values())}")

    print("=== correct ratio ===")
    if total_labeled:
        overall = total_correct / total_labeled
        print(f"  overall: {total_correct}/{total_labeled} = {overall:.3f}")
        for sp in sorted(corr):
            c, n = corr[sp]
            print(f"  {sp}: {c}/{n} = {c / n:.3f}")
        if overall < 0.10 or overall > 0.90:
            print(f"WARNING: correct ratio {overall:.3f} outside 10-90%")
    else:
        print("  no labeled rows")
    print(f"=== null: {total_null} ===")


def run_normal(meta_path, out_path):
    rows = read_meta(meta_path)
    labeled = compute_labels(rows)
    write_labels_csv(labeled, out_path)
    report(labeled)
    print(f"wrote: {out_path}  ({len(labeled)} rows)")


def run_audit(meta_path, n, seed, out_path):
    rows = read_meta(meta_path)
    labeled = compute_labels(rows)
    labeled_only = [(r, correct) for r, correct, _nat in labeled if correct is not None]
    rng = random.Random(seed)
    sample = rng.sample(labeled_only, min(n, len(labeled_only)))
    fields = ["answer", "gold_main", "correct", "manual"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, correct in sample:
            w.writerow({
                "answer": r.get("answer", ""),
                "gold_main": r.get("gold_main", ""),
                "correct": correct,
                "manual": "",
            })
    print(f"=== audit ===")
    print(f"sampled {len(sample)} labeled rows (of {len(labeled_only)}) -> {out_path}")
    print(f"fill the 'manual' column with 0/1, then run: label.py --audit-score {out_path}")


def run_audit_score(audit_path):
    with open(audit_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = 0
    match = 0
    fp = 0  # auto=1, manual=0
    fn = 0  # auto=0, manual=1
    for r in rows:
        manual = (r.get("manual") or "").strip()
        if manual == "":
            continue
        try:
            manual_v = int(manual)
        except ValueError:
            continue
        auto_raw = (r.get("correct") or "").strip()
        if auto_raw == "":
            continue
        auto_v = int(auto_raw)
        total += 1
        if manual_v == auto_v:
            match += 1
        elif auto_v == 1 and manual_v == 0:
            fp += 1
        elif auto_v == 0 and manual_v == 1:
            fn += 1

    print("=== audit score ===")
    print(f"scored rows: {total}  (of {len(rows)} in file)")
    if total:
        print(f"match: {match}/{total} = {match / total:.3f}")
        print(f"false positives (auto=1, manual=0): {fp}")
        print(f"false negatives (auto=0, manual=1): {fn}")
    else:
        print("no scored rows (fill the 'manual' column first)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=str, default=None, help="path to meta.jsonl")
    ap.add_argument("--out", type=str, default=None, help="output CSV path")
    ap.add_argument("--audit", type=int, default=None, help="audit mode: number of rows to sample")
    ap.add_argument("--seed", type=int, default=1, help="seed for audit sampling")
    ap.add_argument("--audit-score", type=str, default=None, help="score a filled audit.csv")
    args = ap.parse_args()

    if args.audit_score:
        run_audit_score(args.audit_score)
        return 0

    if args.audit is not None:
        if not args.meta:
            ap.error("--audit requires --meta")
        out = args.out or "audit.csv"
        run_audit(args.meta, args.audit, args.seed, out)
        return 0

    if not args.meta:
        ap.error("--meta is required")
    out = args.out or "labels.csv"
    run_normal(args.meta, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
