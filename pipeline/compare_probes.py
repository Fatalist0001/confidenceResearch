"""Paired bootstrap comparison of two probe score files.

Labels are model-specific (each model has its own answers), so the resampling unit
is the question: rows are resampled with replacement, and each model keeps its own
outcome and score for that row.

Usage:
    python compare_probes.py --a out/main_qwen08/probe_scores.csv \
        --b out/main_gemma4e2b/probe_scores.csv --split test \
        --out out/compare_qwen08_gemma4e2b_test.json
"""

import argparse
import csv
import json
import sys

import numpy as np

from baseline import auprc, auroc


def load(path, split):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            out[int(r["row"])] = (r["id"], int(r["y"]), float(r["score"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a, b = load(args.a, args.split), load(args.b, args.split)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common rows")

    ya = np.array([a[r][1] for r in common], dtype=float)
    yb = np.array([b[r][1] for r in common], dtype=float)
    sa = np.array([a[r][2] for r in common], dtype=float)
    sb = np.array([b[r][2] for r in common], dtype=float)
    mismatch = int(np.sum(ya != yb))

    point = {
        "n": len(common),
        "pos_a": int(ya.sum()),
        "pos_b": int(yb.sum()),
        "base_rate_a": float(ya.mean()),
        "base_rate_b": float(yb.mean()),
        "label_mismatch_rows": mismatch,
        "a_auroc": auroc(ya, sa),
        "b_auroc": auroc(yb, sb),
        "a_auprc": auprc(ya, sa),
        "b_auprc": auprc(yb, sb),
    }
    point["delta_auroc"] = point["b_auroc"] - point["a_auroc"]
    point["delta_auprc"] = point["b_auprc"] - point["a_auprc"]

    rng = np.random.default_rng(args.seed)
    d_auroc, d_auprc = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(common), len(common))
        if ya[idx].sum() in (0, len(idx)) or yb[idx].sum() in (0, len(idx)):
            continue
        d_auroc.append(auroc(yb[idx], sb[idx]) - auroc(ya[idx], sa[idx]))
        d_auprc.append(auprc(yb[idx], sb[idx]) - auprc(ya[idx], sa[idx]))

    def ci(v):
        lo, hi = np.percentile(np.asarray(v), [2.5, 97.5])
        return [float(lo), float(hi)]

    res = {
        "split": args.split,
        "a": args.a.replace("\\", "/"),
        "b": args.b.replace("\\", "/"),
        "n_boot": args.n_boot,
        "bootstrap_seed": args.seed,
        "metrics": point,
        "delta_auroc_ci": ci(d_auroc),
        "delta_auprc_ci": ci(d_auprc),
        "share_b_higher_auroc": float(np.mean(np.asarray(d_auroc) > 0)),
    }

    print(f"common rows: {point['n']} (base A {point['base_rate_a']:.3f}, "
          f"B {point['base_rate_b']:.3f}; labels differ on {mismatch})")
    print(f"A auroc {point['a_auroc']:.4f} auprc {point['a_auprc']:.4f}")
    print(f"B auroc {point['b_auroc']:.4f} auprc {point['b_auprc']:.4f}")
    print(f"delta AUROC {point['delta_auroc']:+.4f} CI [{res['delta_auroc_ci'][0]:+.4f}, "
          f"{res['delta_auroc_ci'][1]:+.4f}]")
    print(f"delta AUPRC {point['delta_auprc']:+.4f} CI [{res['delta_auprc_ci'][0]:+.4f}, "
          f"{res['delta_auprc_ci'][1]:+.4f}]")
    print(f"share of bootstrap draws with B > A: {res['share_b_higher_auroc']:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
