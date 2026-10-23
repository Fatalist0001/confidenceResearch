"""Cross-model check of probe scores against the other model's labels.

The probes live in different feature spaces (D differs), so their weights cannot
be transferred. What can be compared without any alignment is whether one
model's score ranks the other model's answers correctly:

    AUROC(score_Gemma, label_Qwen)   and   AUROC(score_Qwen, label_Gemma)

Restricted to rows that both runs scored in the same split (same questions,
different answers), with own-model AUROCs recomputed on the same subset.

Usage:
    python cross_probe_scores.py \
        --scores-a out/main_qwen08/probe_scores.csv --labels-a out/main_qwen08/labels.csv \
        --scores-b out/main_gemma4e2b/probe_scores.csv --labels-b out/main_gemma4e2b/labels.csv \
        --split test --out out/cross_qwen08_gemma4e2b_test.json
"""

import argparse
import csv
import json
import sys

import numpy as np

from baseline import auroc


def read_labels(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            v = r["correct_filtered_b"].strip()
            out[int(r["row"])] = None if v == "" else int(v)
    return out


def read_scores(path, split):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            out[int(r["row"])] = (float(r["score"]), int(r["y"]))
    return out


def ci(y, s, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy, ss = y[i], s[i]
        if yy.min() == yy.max():
            continue
        vals.append(auroc(yy, ss))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-a", required=True)
    ap.add_argument("--labels-a", required=True)
    ap.add_argument("--scores-b", required=True)
    ap.add_argument("--labels-b", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    la = read_labels(args.labels_a)
    lb = read_labels(args.labels_b)
    sa = read_scores(args.scores_a, args.split)
    sb = read_scores(args.scores_b, args.split)

    rows = sorted(set(sa) & set(sb))
    rows = [r for r in rows if la[r] is not None and lb[r] is not None]
    if not rows:
        print("no common rows", file=sys.stderr)
        return 1

    ya = np.array([la[r] for r in rows], dtype=float)
    yb = np.array([lb[r] for r in rows], dtype=float)
    a = np.array([sa[r][0] for r in rows], dtype=float)
    b = np.array([sb[r][0] for r in rows], dtype=float)

    for r in rows:
        assert sa[r][1] == la[r] and sb[r][1] == lb[r], "score file label mismatch at row %d" % r

    res = {
        "split": args.split,
        "n_common": len(rows),
        "positives": {"a": int(ya.sum()), "b": int(yb.sum())},
        "base_rate": {"a": float(ya.mean()), "b": float(yb.mean())},
        "own": {
            "a_on_a": {"auroc": auroc(ya, a), "ci": ci(ya, a, args.n_boot, args.seed)},
            "b_on_b": {"auroc": auroc(yb, b), "ci": ci(yb, b, args.n_boot, args.seed)},
        },
        "cross": {
            "b_on_a": {"auroc": auroc(ya, b), "ci": ci(ya, b, args.n_boot, args.seed)},
            "a_on_b": {"auroc": auroc(yb, a), "ci": ci(yb, a, args.n_boot, args.seed)},
        },
        "corr_scores_pearson": float(np.corrcoef(a, b)[0, 1]),
        "label_agreement": float(np.mean(ya == yb)),
        "cross_label_control": {
            "b_label_from_a_label": auroc(yb, ya),
            "a_label_from_b_label": auroc(ya, yb),
        },
        "n_boot": args.n_boot,
        "bootstrap_seed": args.seed,
        "scores_a": args.scores_a.replace("\\", "/"),
        "scores_b": args.scores_b.replace("\\", "/"),
        "labels_a": args.labels_a.replace("\\", "/"),
        "labels_b": args.labels_b.replace("\\", "/"),
        "note": "a = first model, b = second; cross = score of one model vs the other model's label",
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("split %s, common rows %d (a+ %d, b+ %d)" % (args.split, len(rows), int(ya.sum()), int(yb.sum())))
    for k, v in (("own a", res["own"]["a_on_a"]), ("own b", res["own"]["b_on_b"]),
                 ("b on a", res["cross"]["b_on_a"]), ("a on b", res["cross"]["a_on_b"])):
        print("  %-7s AUROC %.4f [%.3f, %.3f]" % (k, v["auroc"], v["ci"][0], v["ci"][1]))
    print("  Pearson(score_a, score_b) %.4f" % res["corr_scores_pearson"])
    print("  labels agree %.3f | label_b ~ label_a %.4f | label_a ~ label_b %.4f"
          % (res["label_agreement"], res["cross_label_control"]["b_label_from_a_label"],
             res["cross_label_control"]["a_label_from_b_label"]))
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
