"""Label correctness of llama-hidden-states outputs against gold answers.

Normal mode:  label.py --meta <out_dir>/meta.jsonl --out labels.csv
Audit mode:   label.py --meta <out_dir>/meta.jsonl --audit N --seed S --questions data/main.jsonl --out audit.csv
              (writes N random sample rows first, then appends strict!=relaxed disagreements)
Score mode:   label.py --audit-score audit.csv [--sample-n 100]
              (splits rows into (a) leading sample and (b) the rest; noise is estimated from (a))
"""

import argparse
import csv
import json
import random
import re
import string
import sys
from collections import defaultdict

from normalize import normalize

_FALLBACK_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_FALLBACK_WHITESPACE_RE = re.compile(r"\s+")


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def normalize_fallback(s):
    """Lowercase, drop punctuation, collapse whitespace — but keep articles."""
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_FALLBACK_PUNCT_TABLE)
    s = _FALLBACK_WHITESPACE_RE.sub(" ", s).strip()
    return s


def norm_with_fallback(s):
    """normalize(s); if that is empty, fall back to normalize_fallback(s)."""
    n = normalize(s)
    if n:
        return n
    return normalize_fallback(s)


def phrase_match(ans_tokens, gold_tokens):
    """True if gold_tokens appears in ans as a contiguous token sequence."""
    gl = len(gold_tokens)
    if gl == 0 or gl > len(ans_tokens):
        return False
    for i in range(len(ans_tokens) - gl + 1):
        if ans_tokens[i:i + gl] == gold_tokens:
            return True
    return False


def all_words_present(gold_tokens, ans_tokens):
    """True if every gold token appears as a whole word in ans (order-free)."""
    ans_set = set(ans_tokens)
    return all(g in ans_set for g in gold_tokens)


def strip_plural(token):
    """Drop a trailing 's' from words of length >= 4 (simple plural normalization)."""
    if len(token) >= 4 and token.endswith("s"):
        return token[:-1]
    return token


def strip_plural_tokens(tokens):
    return [strip_plural(t) for t in tokens]


def relaxed_match(ans_tokens, gold_tokens):
    """strict OR order-free word-subset (gold>=2) OR plural-normalized phrase match."""
    if phrase_match(ans_tokens, gold_tokens):
        return True
    if len(gold_tokens) >= 2 and all_words_present(gold_tokens, ans_tokens):
        return True
    if phrase_match(strip_plural_tokens(ans_tokens), strip_plural_tokens(gold_tokens)):
        return True
    return False


def label_row(row):
    """Return (correct_strict, correct_relaxed, n_answer_tokens, matched_strict, matched_relaxed).

    matched_* is the gold element that gave the match (empty if no match).
    """
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
        return None, None, n_answer_tokens, "", ""
    if truncated:
        return None, None, n_answer_tokens, "", ""
    ans_norm = norm_with_fallback(answer)
    if not ans_norm:
        return None, None, n_answer_tokens, "", ""
    ans_tokens = ans_norm.split()

    strict = 0
    relaxed = 0
    m_strict = ""
    m_relaxed = ""
    for g in gold:
        g_norm = norm_with_fallback(g)
        if not g_norm:
            continue
        g_tokens = g_norm.split()
        if not m_strict and phrase_match(ans_tokens, g_tokens):
            strict = 1
            m_strict = g_norm
        if not m_relaxed and relaxed_match(ans_tokens, g_tokens):
            relaxed = 1
            m_relaxed = g_norm
    return strict, relaxed, n_answer_tokens, m_strict, m_relaxed


def question_phrase_hit(question_norm, matched_norm):
    """True if matched_norm appears in question_norm as a whole (contiguous) phrase."""
    if not question_norm or not matched_norm:
        return False
    return phrase_match(question_norm.split(), matched_norm.split())


def filtered_labels(strict, matched_strict, gold_main, question_norm):
    """Return (correct_filtered_a, correct_filtered_b).

    a: correct_strict, but 0 when matched_strict occurs in the question (whole phrase).
    b: same, but the filter applies only when matched_strict != gold_main (normalized).
    """
    if strict is None:
        return None, None
    if strict == 0:
        return 0, 0
    in_q = question_phrase_hit(question_norm, matched_strict)
    gold_main_norm = norm_with_fallback(gold_main)
    fa = 0 if in_q else 1
    fb = 0 if (in_q and matched_strict != gold_main_norm) else 1
    return fa, fb


def read_meta(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


QUESTION_SUFFIX = "\nAnswer with just the answer, no explanation."


def load_questions(path):
    """Map id -> original question (without the generation prompt suffix)."""
    q = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                s = r.get("question") or ""
                if s.endswith(QUESTION_SUFFIX):
                    s = s[: -len(QUESTION_SUFFIX)]
                q[r.get("id")] = s
    return q


def compute_labels(rows):
    """Return list of (row, correct_strict, correct_relaxed, n_answer_tokens, m_strict, m_relaxed)."""
    return [(r, *label_row(r)) for r in rows]


def write_labels_csv(labeled, out_path, qmap=None):
    fields = ["id", "row", "split", "correct_strict", "correct_relaxed",
              "correct_filtered_a", "correct_filtered_b", "correct",
              "truncated", "n_answer_tokens", "answer", "gold_main"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, strict, relaxed, nat, ms, _mr in labeled:
            qnorm = norm_with_fallback(qmap.get(r.get("id"), "")) if qmap else ""
            fa, fb = filtered_labels(strict, ms, r.get("gold_main"), qnorm)
            w.writerow({
                "id": r.get("id", ""),
                "row": r.get("row", ""),
                "split": r.get("split", ""),
                "correct_strict": "" if strict is None else strict,
                "correct_relaxed": "" if relaxed is None else relaxed,
                "correct_filtered_a": "" if fa is None else fa,
                "correct_filtered_b": "" if fb is None else fb,
                "correct": "" if strict is None else strict,
                "truncated": r.get("truncated", ""),
                "n_answer_tokens": "" if nat is None else nat,
                "answer": r.get("answer", ""),
                "gold_main": r.get("gold_main", ""),
            })


def report(labeled):
    by_split = defaultdict(int)
    strict_corr = defaultdict(lambda: [0, 0])
    relaxed_corr = defaultdict(lambda: [0, 0])
    total_strict = 0
    total_relaxed = 0
    total_labeled = 0
    total_null = 0
    n_diff = 0
    for r, strict, relaxed, _nat, _ms, _mr in labeled:
        sp = r.get("split", "?")
        by_split[sp] += 1
        if strict is None:
            total_null += 1
        else:
            total_labeled += 1
            total_strict += strict
            total_relaxed += relaxed
            strict_corr[sp][0] += strict
            strict_corr[sp][1] += 1
            relaxed_corr[sp][0] += relaxed
            relaxed_corr[sp][1] += 1
            if strict != relaxed:
                n_diff += 1

    print("=== rows by split ===")
    for sp in sorted(by_split):
        print(f"  {sp}: {by_split[sp]}")
    print(f"  total: {sum(by_split.values())}")

    print("=== correct ratio ===")
    if total_labeled:
        ov_s = total_strict / total_labeled
        ov_r = total_relaxed / total_labeled
        print(f"  overall strict : {total_strict}/{total_labeled} = {ov_s:.3f}")
        print(f"  overall relaxed: {total_relaxed}/{total_labeled} = {ov_r:.3f}")
        for sp in sorted(strict_corr):
            cs, ns = strict_corr[sp]
            cr, nr = relaxed_corr[sp]
            print(f"  {sp}: strict {cs}/{ns} = {cs / ns:.3f}   relaxed {cr}/{nr} = {cr / nr:.3f}")
        if ov_s < 0.10 or ov_s > 0.90:
            print(f"WARNING: strict ratio {ov_s:.3f} outside 10-90%")
    else:
        print("  no labeled rows")
    print(f"=== null: {total_null} ===")
    print(f"=== strict != relaxed: {n_diff} rows ===")


def run_normal(meta_path, out_path, questions_path="data/main.jsonl"):
    rows = read_meta(meta_path)
    labeled = compute_labels(rows)
    qmap = load_questions(questions_path) if questions_path else {}
    write_labels_csv(labeled, out_path, qmap)
    report(labeled)
    print(f"wrote: {out_path}  ({len(labeled)} rows)")


def run_audit(meta_path, n, seed, out_path, questions_path):
    rows = read_meta(meta_path)
    labeled = compute_labels(rows)
    qmap = load_questions(questions_path)
    labeled_only = [(r, strict, relaxed, ms, mr) for r, strict, relaxed, _nat, ms, mr in labeled
                    if strict is not None]
    rng = random.Random(seed)
    sample = rng.sample(labeled_only, min(n, len(labeled_only)))
    sample_ids = {r.get("id") for r, _s, _rl, _ms, _mr in sample}
    # strict != relaxed disagreements, in meta order, excluding rows already sampled
    diff = [(r, strict, relaxed, ms, mr) for r, strict, relaxed, _nat, ms, mr in labeled
            if strict is not None and strict != relaxed and r.get("id") not in sample_ids]

    fields = ["id", "question", "answer", "gold_main", "correct_strict", "correct_relaxed",
              "matched_strict", "matched_relaxed", "manual"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, strict, relaxed, ms, mr in sample + diff:
            w.writerow({
                "id": r.get("id", ""),
                "question": qmap.get(r.get("id"), ""),
                "answer": r.get("answer", ""),
                "gold_main": r.get("gold_main", ""),
                "correct_strict": strict,
                "correct_relaxed": relaxed,
                "matched_strict": ms,
                "matched_relaxed": mr,
                "manual": "",
            })
    print(f"=== audit ===")
    print(f"sampled {len(sample)} labeled rows (of {len(labeled_only)}) + {len(diff)} disagreements")
    print(f"rows: {len(sample) + len(diff)} -> {out_path}")
    print(f"fill the 'manual' column with 0/1, then run: label.py --audit-score {out_path}")


def _score_group(rows):
    stats = {}
    for label_col in ("correct_strict", "correct_relaxed"):
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
            auto_raw = (r.get(label_col) or "").strip()
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
        noise = (fp + fn) / total if total else 0.0
        stats[label_col] = {"total": total, "match": match, "fp": fp, "fn": fn, "noise": noise}
    return stats


def _print_group(name, rows, stats):
    print(f"--- {name}: {len(rows)} rows ---")
    for label_col in ("correct_strict", "correct_relaxed"):
        s = stats[label_col]
        if s["total"]:
            print(f"  {label_col}:")
            print(f"    match: {s['match']}/{s['total']} = {s['match'] / s['total']:.3f}")
            print(f"    false positives (auto=1, manual=0): {s['fp']}")
            print(f"    false negatives (auto=0, manual=1): {s['fn']}")
            print(f"    noise: {s['noise']:.3f}")
        else:
            print(f"  {label_col}: no scored rows")
    ov = (stats["correct_strict"]["noise"] + stats["correct_relaxed"]["noise"]) / 2
    print(f"  overall noise (mean of two): {ov:.3f}")


def run_audit_score(audit_path, sample_n=100):
    with open(audit_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # (a) first sample_n rows = random sample; (b) the rest = selected disagreements
    group_a = rows[:sample_n]
    group_b = rows[sample_n:]

    print("=== audit score ===")
    print(f"rows in file: {len(rows)}")
    stats_a = _score_group(group_a)
    stats_b = _score_group(group_b)
    _print_group(f"(a) random sample (rows 1-{min(sample_n, len(rows))})", group_a, stats_a)
    _print_group(f"(b) disagreements (rows {min(sample_n, len(rows)) + 1}-{len(rows)})", group_b, stats_b)

    print("=== noise estimate (from (a) random sample) ===")
    for label_col in ("correct_strict", "correct_relaxed"):
        s = stats_a[label_col]
        if s["total"]:
            print(f"  {label_col}: {s['noise']:.3f}  ({s['fp']} fp + {s['fn']} fn / {s['total']})")
        else:
            print(f"  {label_col}: no scored rows in (a)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=str, default=None, help="path to meta.jsonl")
    ap.add_argument("--out", type=str, default=None, help="output CSV path")
    ap.add_argument("--audit", type=int, default=None, help="audit mode: number of rows to sample")
    ap.add_argument("--seed", type=int, default=1, help="seed for audit sampling")
    ap.add_argument("--questions", type=str, default="data/main.jsonl",
                    help="path to original questions jsonl (audit id/question columns; labels filter)")
    ap.add_argument("--audit-score", type=str, default=None, help="score a filled audit.csv")
    ap.add_argument("--sample-n", type=int, default=100,
                    help="audit-score: number of leading rows treated as the random sample")
    args = ap.parse_args()

    if args.audit_score:
        run_audit_score(args.audit_score, args.sample_n)
        return 0

    if args.audit is not None:
        if not args.meta:
            ap.error("--audit requires --meta")
        out = args.out or "audit.csv"
        run_audit(args.meta, args.audit, args.seed, out, args.questions)
        return 0

    if not args.meta:
        ap.error("--meta is required")
    out = args.out or "labels.csv"
    run_normal(args.meta, out, args.questions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
