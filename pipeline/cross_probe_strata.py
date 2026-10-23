"""Stratified cross-model check: does A's score add anything beyond A's outcome?

The plain cross AUROC (cross_probe_scores.py) mixes two things: how hard the
question is for both models, and any extra signal in A's score.  Since both
models answer the same questions, difficulty alone already produces AUROC > 0.5.
Here the rows are split by A's label and AUROC(score_A, label_B) is computed
inside each stratum, i.e. where "A was right/wrong" is held constant.

The logistic check fits on one split (default val) and evaluates on the other
(default test): how much does adding score_A improve on label_A alone as a
predictor of label_B?

Usage:
    python cross_probe_strata.py \
        --scores-a out/main_qwen08/probe_scores.csv --labels-a out/main_qwen08/labels.csv \
        --scores-b out/main_gemma4e2b/probe_scores.csv --labels-b out/main_gemma4e2b/labels.csv \
        --split test --out out/cross_strata_qwen08_gemma4e2b_test.json
"""

import argparse
import json
import sys

import numpy as np

from baseline import auroc
from cross_probe_scores import ci, read_labels, read_scores
from probe import fit_l2, with_bias


def load(scores_a, labels_a, scores_b, labels_b, split):
    sa = read_scores(scores_a, split)
    sb = read_scores(scores_b, split)
    la, lb = read_labels(labels_a), read_labels(labels_b)
    rows = [r for r in sorted(set(sa) & set(sb)) if la[r] is not None and lb[r] is not None]
    for r in rows:
        assert sa[r][1] == la[r] and sb[r][1] == lb[r], "score file label mismatch at row %d" % r
    ya = np.array([la[r] for r in rows], dtype=float)
    yb = np.array([lb[r] for r in rows], dtype=float)
    a = np.array([sa[r][0] for r in rows], dtype=float)
    return ya, yb, a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-a", required=True)
    ap.add_argument("--labels-a", required=True)
    ap.add_argument("--scores-b", required=True)
    ap.add_argument("--labels-b", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ya, yb, a = load(args.scores_a, args.labels_a, args.scores_b, args.labels_b, args.split)
    if len(ya) == 0:
        print("no common rows", file=sys.stderr)
        return 1

    strata = {}
    weights = []
    for val, name in ((1.0, "a_correct"), (0.0, "a_wrong")):
        m = ya == val
        yy, ss = yb[m], a[m]
        if len(yy) == 0:
            continue
        auc = auroc(yy, ss) if yy.min() != yy.max() else float("nan")
        strata[name] = {
            "n": int(m.sum()),
            "positives": int(yy.sum()),
            "base_rate": float(yy.mean()),
            "auroc": auc,
            "ci": ci(yy, ss, args.n_boot, args.seed),
        }
        if not np.isnan(auc):
            weights.append((len(yy), auc))

    cond = float(sum(w * v for w, v in weights) / sum(w for w, _ in weights)) if weights else float("nan")

    fit = None
    if args.fit_split != args.split:
        fya, fyb, fa = load(args.scores_a, args.labels_a, args.scores_b, args.labels_b,
                            args.fit_split)
        mu, sd = fa.mean(), fa.std() + 1e-12
        zf, ze = (fa - mu) / sd, (a - mu) / sd
        w0 = fit_l2(fya[:, None], fyb, 1e-4)
        w1 = fit_l2(np.column_stack([fya, zf]), fyb, 1e-4)
        auc0 = auroc(yb, with_bias(ya[:, None]) @ w0)
        auc1 = auroc(yb, with_bias(np.column_stack([ya, ze])) @ w1)
        fit = {
            "fit_split": args.fit_split,
            "eval_split": args.split,
            "n_fit": len(fya),
            "auroc_label_a_only": auc0,
            "auroc_label_a_plus_score_a": auc1,
            "delta": auc1 - auc0,
        }

    res = {
        "split": args.split,
        "n": len(ya),
        "positives": {"a": int(ya.sum()), "b": int(yb.sum())},
        "base_rate": {"a": float(ya.mean()), "b": float(yb.mean())},
        "overall_cross_auroc": auroc(yb, a),
        "strata": strata,
        "conditional_auroc_weighted": cond,
        "logistic": fit,
        "n_boot": args.n_boot,
        "bootstrap_seed": args.seed,
        "scores_a": args.scores_a.replace("\\", "/"),
        "scores_b": args.scores_b.replace("\\", "/"),
        "note": "cross AUROC within strata of A's label; logistic fits on fit_split",
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("split %s, rows %d (a+ %d, b+ %d)" % (args.split, len(ya), int(ya.sum()), int(yb.sum())))
    print("  overall cross %.4f | conditional (within label_a) %.4f" % (res["overall_cross_auroc"], cond))
    for name, s in strata.items():
        print("  %-9s n=%4d b+ %3d base %.2f  AUROC %.4f [%.3f, %.3f]"
              % (name, s["n"], s["positives"], s["base_rate"], s["auroc"], s["ci"][0], s["ci"][1]))
    if fit:
        print("  logistic (fit %s, eval %s): label_a %.4f -> +score_a %.4f (delta %+.4f)"
              % (fit["fit_split"], fit["eval_split"], fit["auroc_label_a_only"],
                 fit["auroc_label_a_plus_score_a"], fit["delta"]))
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
