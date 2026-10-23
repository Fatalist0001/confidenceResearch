"""Probe: L2 logistic regression on hidden states, per point (A/C) and layer.

Protocol (frozen before touching test):
  * features: hidden state vector at layer L, point A or C; standardized on train;
  * L2 strength lambda is chosen from a fixed grid by the MEAN val AUROC across all
    48 (point, layer) configs;
  * final config is chosen by max val AUROC at that lambda;
  * test metrics are computed once, for the selected config only.
Controls: permuted train labels (several permutations) and layer 0.

Usage:
    python probe.py --out out/main_qwen08/probe.json \
        --scores-out out/main_qwen08/probe_scores.csv \
        --curve-out out/main_qwen08/probe_curve.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from baseline import auroc, auprc, fit_logistic, metrics, sigmoid


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["row"]): r for r in csv.DictReader(f)}


def split_indices(meta, labels):
    out = {}
    for i, rec in enumerate(meta):
        lab = labels[int(rec["row"])]
        if rec.get("status") != "ok" or not rec.get("generated") or rec.get("truncated"):
            continue
        if lab["correct_filtered_b"].strip() == "":
            continue
        out.setdefault(rec["split"], []).append(i)
    return out


def fit_l2(X, y, lam, max_iter=40, max_step=2.0, tol=1e-6):
    """Newton (IRLS) for logistic loss + lam/2 * ||w||^2, step norm capped."""
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])
    reg = np.ones(d + 1)
    reg[0] = 0.0
    w = np.zeros(d + 1)
    for it in range(max_iter):
        p = sigmoid(Xb @ w)
        W = np.maximum(p * (1 - p), 1e-6)
        g = Xb.T @ (y - p) - lam * (reg * w)
        H = (Xb * W[:, None]).T @ Xb + lam * np.diag(reg)
        step = np.linalg.solve(H, g)
        norm = float(np.linalg.norm(step))
        if norm > max_step:
            step *= max_step / norm
            norm = max_step
        w += step
        if norm < tol:
            break
    return w


def standardize(train, others):
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1e-8
    return (train - mu) / sd, [((x - mu) / sd) for x in others]


def with_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def save_probe_model(path, w, mu, sd, info):
    """Store the fitted probe as a portable object (no pickled code, plain arrays)."""
    np.savez(
        path,
        w=np.asarray(w, dtype=np.float64),
        mu=np.asarray(mu, dtype=np.float64),
        sd=np.asarray(sd, dtype=np.float64),
        meta=np.array(json.dumps(info, ensure_ascii=False)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-a", default="out/main_qwen08/hidden_a.npy")
    ap.add_argument("--hidden-c", default="out/main_qwen08/hidden_c.npy")
    ap.add_argument("--meta", default="out/main_qwen08/meta.jsonl")
    ap.add_argument("--labels", default="out/main_qwen08/labels.csv")
    ap.add_argument("--out", default="out/main_qwen08/probe.json")
    ap.add_argument("--scores-out", default="out/main_qwen08/probe_scores.csv")
    ap.add_argument("--curve-out", default="out/main_qwen08/probe_curve.csv")
    ap.add_argument("--model-out", default=None,
                    help="weights of the selected probe (.npz); default: probe_model.npz next to --out")
    ap.add_argument("--lambdas", default="100,300,1000,3000")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffles", type=int, default=5)
    args = ap.parse_args()

    meta = read_jsonl(args.meta)
    labels = read_labels(args.labels)
    idx = split_indices(meta, labels)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    ytr = np.array([int(labels[meta[i]["row"]]["correct_filtered_b"]) for i in tr], dtype=float)
    yva = np.array([int(labels[meta[i]["row"]]["correct_filtered_b"]) for i in va], dtype=float)
    yte = np.array([int(labels[meta[i]["row"]]["correct_filtered_b"]) for i in te], dtype=float)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    print(f"rows: train {len(tr)} ({int(ytr.sum())}+), val {len(va)} ({int(yva.sum())}+), "
          f"test {len(te)} ({int(yte.sum())}+)")

    curve = []
    for point, path in (("A", args.hidden_a), ("C", args.hidden_c)):
        arr = np.load(path, mmap_mode="r")
        n_layers = arr.shape[1]
        for layer in range(n_layers):
            Xtr = np.asarray(arr[tr, layer, :], dtype=np.float64)
            Xva = np.asarray(arr[va, layer, :], dtype=np.float64)
            Xte = np.asarray(arr[te, layer, :], dtype=np.float64)
            Ztr, (Zva, Zte) = standardize(Xtr, [Xva, Xte])
            Btr, Bva, Bte = with_bias(Ztr), with_bias(Zva), with_bias(Zte)
            for lam in lambdas:
                w = fit_l2(Ztr, ytr, lam)
                s_tr, s_va = Btr @ w, Bva @ w
                curve.append(
                    {
                        "point": point,
                        "layer": layer,
                        "lambda": lam,
                        "train_auroc": auroc(ytr, s_tr),
                        "val_auroc": auroc(yva, s_va),
                        "val_auprc": auprc(yva, s_va),
                    }
                )
        print(f"point {point}: {n_layers} layers x {len(lambdas)} lambdas done")

    lam_mean = {}
    for lam in lambdas:
        vals = [c["val_auroc"] for c in curve if c["lambda"] == lam]
        lam_mean[lam] = float(np.mean(vals))
    lam_best = max(lambdas, key=lambda l: lam_mean[l])
    print("mean val AUROC by lambda: " + ", ".join(f"{l:g}:{lam_mean[l]:.4f}" for l in lambdas))
    print(f"selected lambda: {lam_best:g}")

    cand = [c for c in curve if c["lambda"] == lam_best]
    sel = max(cand, key=lambda c: c["val_auroc"])
    print(f"selected config: point {sel['point']}, layer {sel['layer']} "
          f"(val AUROC {sel['val_auroc']:.4f}, val AUPRC {sel['val_auprc']:.4f})")

    # refit the selected config on train (same lambda, same standardization)
    arr = np.load(args.hidden_a if sel["point"] == "A" else args.hidden_c, mmap_mode="r")
    Xtr = np.asarray(arr[tr, sel["layer"], :], dtype=np.float64)
    Xva = np.asarray(arr[va, sel["layer"], :], dtype=np.float64)
    Xte = np.asarray(arr[te, sel["layer"], :], dtype=np.float64)
    Ztr, (Zva, Zte) = standardize(Xtr, [Xva, Xte])
    Btr, Bva, Bte = with_bias(Ztr), with_bias(Zva), with_bias(Zte)
    w = fit_l2(Ztr, ytr, lam_best)
    s_va, s_te = Bva @ w, Bte @ w
    p_va, p_te = sigmoid(s_va), sigmoid(s_te)

    # same mean/std as standardize() above, kept for reuse on other data
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    model_out = args.model_out or os.path.join(os.path.dirname(args.out) or ".", "probe_model.npz")
    save_probe_model(
        model_out, w, mu, sd,
        {
            "version": 1,
            "kind": "l2_logistic_probe",
            "point": sel["point"],
            "layer": int(sel["layer"]),
            "lambda": lam_best,
            "label_column": "correct_filtered_b",
            "w_layout": "[intercept, coef[D]]",
            "apply": "z = (x - mu) / sd; score = w[0] + w[1:] @ z",
            "standardization": "mean/std over the train split",
            "hidden": (args.hidden_a if sel["point"] == "A" else args.hidden_c).replace("\\", "/"),
            "meta": args.meta.replace("\\", "/"),
            "labels": args.labels.replace("\\", "/"),
            "rows": {"train": len(tr), "val": len(va), "test": len(te)},
            "positives": {"train": int(ytr.sum()), "val": int(yva.sum()), "test": int(yte.sum())},
        },
    )
    print(f"saved probe model: {model_out}")

    b0, b1 = fit_logistic(s_va, yva)
    p_va_platt = sigmoid(b0 + b1 * s_va)
    p_te_platt = sigmoid(b0 + b1 * s_te)

    block = {
        "val": metrics(yva, s_va, p_va, args.bins, args.n_boot, args.seed),
        "test": metrics(yte, s_te, p_te, args.bins, args.n_boot, args.seed),
        "test_platt": metrics(yte, s_te, p_te_platt, args.bins, args.n_boot, args.seed),
        "platt": {"intercept": b0, "beta": b1},
    }

    # controls: permuted train labels, same config and lambda
    rng = np.random.default_rng(args.seed)
    shuf = {"val_auroc": [], "test_auroc": [], "val_auprc": []}
    for _ in range(args.shuffles):
        yp = rng.permutation(ytr)
        w = fit_l2(Ztr, yp, lam_best)
        shuf["val_auroc"].append(auroc(yva, Bva @ w))
        shuf["val_auprc"].append(auprc(yva, Bva @ w))
        shuf["test_auroc"].append(auroc(yte, Bte @ w))
    shuf["test_auroc_mean"] = float(np.mean(shuf["test_auroc"]))
    shuf["test_auroc_sd"] = float(np.std(shuf["test_auroc"], ddof=1)) if len(shuf["test_auroc"]) > 1 else 0.0

    l0 = [c for c in curve if c["lambda"] == lam_best and c["layer"] == 0]
    layer0 = {}
    for c in l0:
        arr = np.load(args.hidden_a if c["point"] == "A" else args.hidden_c, mmap_mode="r")
        Xtr0 = np.asarray(arr[tr, 0, :], dtype=np.float64)
        Xva0 = np.asarray(arr[va, 0, :], dtype=np.float64)
        Xte0 = np.asarray(arr[te, 0, :], dtype=np.float64)
        Ztr0, (Zva0, Zte0) = standardize(Xtr0, [Xva0, Xte0])
        w0 = fit_l2(Ztr0, ytr, lam_best)
        layer0[c["point"]] = {
            "val_auroc": auroc(yva, with_bias(Zva0) @ w0),
            "test_auroc": auroc(yte, with_bias(Zte0) @ w0),
            "val_auprc": auprc(yva, with_bias(Zva0) @ w0),
        }

    results = {
        "meta": args.meta.replace("\\", "/"),
        "labels": args.labels.replace("\\", "/"),
        "label_column": "correct_filtered_b",
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "positives": {"train": int(ytr.sum()), "val": int(yva.sum()), "test": int(yte.sum())},
        "lambdas": lambdas,
        "lambda_mean_val_auroc": {f"{k:g}": v for k, v in lam_mean.items()},
        "selected_lambda": lam_best,
        "selection_rule": "lambda: max mean val AUROC over all configs; config: max val AUROC",
        "selected": {
            "point": sel["point"],
            "layer": int(sel["layer"]),
            "val_auroc": sel["val_auroc"],
            "val_auprc": sel["val_auprc"],
        },
        "metrics": block,
        "controls": {"shuffled_labels": shuf, "layer0": layer0},
        "n_boot": args.n_boot,
        "bins": args.bins,
        "bootstrap_seed": args.seed,
        "curve": curve,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(args.curve_out, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        wr.writeheader()
        wr.writerows(curve)
    with open(args.scores_out, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "row", "split", "y", "score", "prob", "prob_platt"])
        for split, rows, y, s, p in (
            ("val", va, yva, s_va, p_va_platt),
            ("test", te, yte, s_te, p_te_platt),
        ):
            for i, yy, ss, pp in zip(rows, y, s, p):
                wr.writerow(
                    [
                        meta[i]["id"],
                        meta[i]["row"],
                        split,
                        int(yy),
                        f"{ss:.9g}",
                        f"{sigmoid(ss):.9g}",
                        f"{pp:.9g}",
                    ]
                )

    print()
    print(f"{'split':5} {'AUROC':>8} {'AUPRC':>8} {'Brier':>7} {'ECE':>6}  AUROC 95% CI")
    for name in ("val", "test", "test_platt"):
        m = block[name]
        ci = m.get("auroc_ci") or [float("nan"), float("nan")]
        print(f"{name:11} {m['auroc']:.4f} {m['auprc']:.4f} {m['brier']:.4f} {m['ece']:.4f} "
              f" [{ci[0]:.3f}, {ci[1]:.3f}]")
    print()
    print(f"control shuffled labels: test AUROC {shuf['test_auroc_mean']:.3f} "
          f"± {shuf['test_auroc_sd']:.3f} over {args.shuffles} permutations")
    for pt in ("A", "C"):
        if pt in layer0:
            print(f"control layer 0 ({pt}): val AUROC {layer0[pt]['val_auroc']:.3f}, "
                  f"test AUROC {layer0[pt]['test_auroc']:.3f}")
    print(f"written: {args.out}")
    print(f"written: {args.curve_out}")
    print(f"written: {args.scores_out}")
    print(f"written: {model_out}")


if __name__ == "__main__":
    sys.exit(main())
