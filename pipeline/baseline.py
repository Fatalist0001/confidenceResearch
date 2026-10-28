"""Baseline: first-answer-token logprob and entropy as correctness confidence.

Reads meta.jsonl (logprobs/entropies per answer token) and labels.csv
(correct_filtered_b), fits a 1-feature logistic calibration on the val split,
and reports AUROC/AUPRC/Brier/ECE with bootstrap intervals on val and test.

Usage:
    python baseline.py --meta out/main_qwen08/meta.jsonl \
        --labels out/main_qwen08/labels.csv \
        --out out/main_qwen08/baseline.json \
        --scores-out out/main_qwen08/baseline_scores.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


def norm_path(p):
    """Normalize separators so reruns with / or \\ produce identical JSON."""
    return os.path.normpath(p).replace("\\", "/")


def read_meta(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON ({e}); interrupted run? see verify_run.py")
    return out


def read_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["row"]): r for r in csv.DictReader(f)}


def build_rows(meta, labels):
    rows = []
    skipped = {"status": 0, "not_generated": 0, "truncated": 0, "no_label": 0, "no_meta": 0}
    for rec in meta:
        lab = labels.get(int(rec["row"]))
        if rec.get("status") != "ok":
            skipped["status"] += 1
            continue
        if not rec.get("generated"):
            skipped["not_generated"] += 1
            continue
        if rec.get("truncated"):
            skipped["truncated"] += 1
            continue
        if lab is None or lab["correct_filtered_b"].strip() == "":
            skipped["no_label"] += 1
            continue
        lps, hs = rec.get("logprobs") or [], rec.get("entropies") or []
        if not lps or not hs:
            skipped["no_meta"] += 1
            continue
        if lab["split"] != rec["split"]:
            raise SystemExit(f"split mismatch at row {rec['row']}: {lab['split']} vs {rec['split']}")
        rows.append(
            {
                "id": rec["id"],
                "row": int(rec["row"]),
                "split": rec["split"],
                "y": int(lab["correct_filtered_b"]),
                "lp0": float(lps[0]),
                "h0": float(hs[0]),
            }
        )
    return rows, skipped


def rankdata(a):
    """Ranks with ties averaged (1-based)."""
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(y, s):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y, s):
    """Average precision, tie-grouped (no interpolation)."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i, n = 0, len(s)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        tp += int(ys[i : j + 1].sum())
        fp += (j - i + 1) - int(ys[i : j + 1].sum())
        recall = tp / n_pos
        precision = tp / (tp + fp) if tp + fp else 0.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return float(ap)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n:
            total += n * abs(float(y[m].mean()) - float(p[m].mean()))
    return total / len(y)


def sigmoid(x):
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(x, y, iters=100, tol=1e-10):
    """IRLS for p = sigmoid(b0 + b1*x); returns (b0, b1)."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = np.clip(sigmoid(X @ w), 1e-9, 1 - 1e-9)
        W = p * (1 - p)
        z = X @ w + (y - p) / W
        sw = np.sqrt(W)
        w_new, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return float(w[0]), float(w[1])


def ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


def metrics(y, s, p=None, bins=10, n_boot=0, seed=0, point=True):
    out = {"n": int(len(y)), "pos": int(y.sum()), "base_rate": float(y.mean())}
    if point:
        out["auroc"] = auroc(y, s)
        out["auprc"] = auprc(y, s)
        if p is not None:
            out["brier"] = brier(y, p)
            out["ece"] = ece(y, p, bins)
    if n_boot:
        rng = np.random.default_rng(seed)
        acc = {"auroc": [], "auprc": [], "brier": [], "ece": []}
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yb, sb = y[idx], s[idx]
            if yb.sum() in (0, len(yb)):
                continue
            acc["auroc"].append(auroc(yb, sb))
            acc["auprc"].append(auprc(yb, sb))
            if p is not None:
                acc["brier"].append(brier(yb, p[idx]))
                acc["ece"].append(ece(yb, p[idx], bins))
        for k, v in acc.items():
            if v:
                out[k + "_ci"] = ci(v)
    return out


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fmt_ci(m, key):
    v = m.get(key)
    c = m.get(key + "_ci")
    if v is None:
        return "n/a"
    if c is None:
        return fmt(v)
    return f"{fmt(v)} [{fmt(c[0])}, {fmt(c[1])}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="out/main_qwen08/meta.jsonl")
    ap.add_argument("--labels", default="out/main_qwen08/labels.csv")
    ap.add_argument("--out", default="out/main_qwen08/baseline.json")
    ap.add_argument("--scores-out", default="out/main_qwen08/baseline_scores.csv")
    ap.add_argument("--calib-split", default="val", help="split used to fit calibration")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, skipped = build_rows(read_meta(args.meta), read_labels(args.labels))
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)

    calib = by_split.get(args.calib_split)
    if not calib:
        raise SystemExit(f"no rows in calibration split '{args.calib_split}'")

    yc = np.array([r["y"] for r in calib], dtype=float)
    coef = {
        "logprob": fit_logistic(-np.array([r["lp0"] for r in calib]), yc),
        "entropy": fit_logistic(-np.array([r["h0"] for r in calib]), yc),
    }

    def coef_str(name):
        b0, b1 = coef[name]
        sign = "x = -lp0" if name == "logprob" else "x = -h0"
        return f"p = sigmoid({b0:.4f} + {b1:.4f} * x), {sign} (fit on {args.calib_split})"

    results = {
        "meta": norm_path(args.meta),
        "labels": norm_path(args.labels),
        "label_column": "correct_filtered_b",
        "calib_split": args.calib_split,
        "n_boot": args.n_boot,
        "bins": args.bins,
        "bootstrap_seed": args.seed,
        "rows": len(rows),
        "skipped": skipped,
        "calibration": {k: {"intercept": v[0], "beta": v[1]} for k, v in coef.items()},
        "splits": {},
    }

    header = (
        f"{'split':6} {'n':>5} {'pos':>4} {'base':>5}  {'score':18}"
        f" {'AUROC [95% CI]':30} {'AUPRC [95% CI]':30} {'Brier [95% CI]':26} {'ECE [95% CI]':24}"
    )
    print(header)
    print("-" * len(header))

    score_rows = []
    for split in ("val", "test"):
        rs = by_split.get(split)
        if not rs:
            continue
        y = np.array([r["y"] for r in rs], dtype=float)
        lp = np.array([r["lp0"] for r in rs], dtype=float)
        h = np.array([r["h0"] for r in rs], dtype=float)
        p_lp = sigmoid(coef["logprob"][0] + coef["logprob"][1] * (-lp))
        p_h = sigmoid(coef["entropy"][0] + coef["entropy"][1] * (-h))

        block = {
            "logprob": metrics(y, lp, None, args.bins, args.n_boot, args.seed),
            "entropy": metrics(y, -h, None, args.bins, args.n_boot, args.seed),
            "logprob_calibrated": metrics(y, p_lp, p_lp, args.bins, args.n_boot, args.seed),
            "entropy_calibrated": metrics(y, p_h, p_h, args.bins, args.n_boot, args.seed),
            "logprob_raw_prob": metrics(y, lp, np.exp(lp), args.bins, args.n_boot, args.seed),
        }
        results["splits"][split] = block

        for name in ("logprob", "entropy", "logprob_calibrated", "entropy_calibrated", "logprob_raw_prob"):
            m = block[name]
            print(
                f"{split:6} {m['n']:5d} {m['pos']:4d} {m['base_rate']:.3f}  {name:18}"
                f" {fmt_ci(m, 'auroc'):30} {fmt_ci(m, 'auprc'):30}"
                f" {fmt_ci(m, 'brier'):26} {fmt_ci(m, 'ece'):24}"
            )
        for r, pl, ph in zip(rs, p_lp, p_h):
            score_rows.append(
                {
                    "id": r["id"],
                    "row": r["row"],
                    "split": r["split"],
                    "y": r["y"],
                    "lp0": f"{r['lp0']:.9g}",
                    "h0": f"{r['h0']:.9g}",
                    "p_raw": f"{np.exp(r['lp0']):.9g}",
                    "p_cal_logprob": f"{pl:.9g}",
                    "p_cal_entropy": f"{ph:.9g}",
                }
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(args.scores_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        w.writeheader()
        w.writerows(score_rows)

    print()
    print("score rows:", len(rows), "skipped:", skipped)
    for split in ("val", "test"):
        if split in results["splits"]:
            b = results["splits"][split]
            print(f"  {split}: {b['logprob']['n']} rows, {b['logprob']['pos']} positive")
    print(f"calibration: {coef_str('logprob')}")
    print(f"calibration: {coef_str('entropy')}")
    print(f"written: {args.out}")
    print(f"written: {args.scores_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
