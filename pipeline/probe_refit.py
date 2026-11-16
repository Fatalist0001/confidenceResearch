"""Refit the probe on target-domain rows instead of transferring frozen weights.

Two variants on identical splits:

    scratch    - new probe fitted from zero on target train rows
    finetune   - source probe (w, mu, sd) continued on target train rows

Reported alongside them, on the same test rows: the frozen source probe and the
first-token baselines. Lambda is chosen on a val slice carved out of train, so the
test rows are scored once per split and never used for selection.

Also reports a k-fold ceiling over all rows and paired differences against the
baselines, so "the signal exists" and "it beats the free baseline" stay separate
claims.

Usage:
    python probe_refit.py --run out/main_qwen08_q4_hotpotqa_v2 \
        --scores out/main_qwen08_q4_hotpotqa_v2/transfer_scores.csv \
        --source-probe out/main_qwen08_q4/probe_model.npz \
        --out out/main_qwen08_q4_hotpotqa_v2/refit_curve.json
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

from baseline import auroc


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_l2(X, y, lam, w0=None, max_iter=40, max_step=2.0, tol=1e-6):
    """Newton (IRLS) for logistic loss + lam/2 * ||w||^2, step norm capped.

    Same optimizer as probe.py; w0 only exists for the warm-start variant.
    """
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])
    reg = np.ones(d + 1)
    reg[0] = 0.0
    w = np.zeros(d + 1) if w0 is None else np.array(w0, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        p = sigmoid(Xb @ w)
        W = np.maximum(p * (1.0 - p), 1e-6)
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


def score(w, X, mu, sd):
    return w[0] + ((X - mu) / sd) @ w[1:]


def stratified_split(y, n_train, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = max(1, min(int(round(n_train * len(pos) / len(y))), len(pos)))
    n_neg = max(1, min(n_train - n_pos, len(neg)))
    tr = np.concatenate([pos[:n_pos], neg[:n_neg]])
    rng.shuffle(tr)
    mask = np.ones(len(y), dtype=bool)
    mask[tr] = False
    return tr, np.flatnonzero(mask)


def inner_val(y_tr, val_frac, rng):
    pos = np.flatnonzero(y_tr == 1)
    neg = np.flatnonzero(y_tr == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    kp = max(1, int(round(val_frac * len(pos))))
    kn = max(1, int(round(val_frac * len(neg))))
    va = np.concatenate([pos[:kp], neg[:kn]])
    rng.shuffle(va)
    mask = np.ones(len(y_tr), dtype=bool)
    mask[va] = False
    return np.flatnonzero(mask), va


def pick_lambda(Xtr, y_tr, rng, lambdas, val_frac, w0=None):
    inner, va = inner_val(y_tr, val_frac, rng)
    best, best_lam = -1.0, lambdas[0]
    for lam in lambdas:
        w = fit_l2(Xtr[inner], y_tr[inner].astype(np.float64), lam, w0=w0)
        a = auroc(y_tr[va].astype(np.float64), score(w, Xtr[va], 0.0, 1.0))
        if np.isfinite(a) and a > best:
            best, best_lam = a, lam
    return best_lam


def fit_and_eval(X, y, tr, te, rng, lambdas, val_frac, mu=None, sd=None, w0=None):
    """Fit on tr (lambda from an inner val slice), return test AUROC and lambda."""
    if mu is None:
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    lam = pick_lambda(Xtr, y[tr], rng, lambdas, val_frac, w0=w0)
    w = fit_l2(Xtr, y[tr].astype(np.float64), lam, w0=w0)
    return auroc(y[te].astype(np.float64), score(w, Xte, 0.0, 1.0)), lam


def load_rows(run, scores_name, label_column, layer_index):
    hidden = os.path.join(run, "hidden_c.npy")
    X = np.load(hidden)[:, layer_index, :].astype(np.float64)
    path = os.path.join(run, scores_name)
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("analysis_included", "1") == "1"]
    idx = np.array([int(r["row"]) for r in rows])
    y = np.array([int(r[label_column]) for r in rows], dtype=np.int64)
    lp = np.array([float(r["first_token_logprob"]) for r in rows])
    ne = np.array([float(r["negative_first_token_entropy"]) for r in rows])
    return X[idx], y, lp, ne, hidden, path


def kfold_cv(X, y, folds, lambdas, val_frac, seed):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    buckets = [[] for _ in range(folds)]
    for arr in (pos, neg):
        for i, v in enumerate(arr):
            buckets[i % folds].append(int(v))
    scores, lams = [], []
    for k in range(folds):
        te = np.array(sorted(buckets[k]))
        mask = np.ones(len(y), dtype=bool)
        mask[te] = False
        tr = np.flatnonzero(mask)
        a, lam = fit_and_eval(X, y, tr, te, np.random.default_rng(seed + 1000 + k),
                              lambdas, val_frac)
        scores.append(a)
        lams.append(lam)
    return {
        "folds": folds,
        "auroc_mean": float(np.mean(scores)),
        "auroc_sd": float(np.std(scores, ddof=1)),
        "auroc_folds": [float(s) for s in scores],
        "lambda_folds": [float(l) for l in lams],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/main_qwen08_q4_hotpotqa_v2")
    ap.add_argument("--scores", default="transfer_scores.csv")
    ap.add_argument("--label-column", default="hotpot_em")
    ap.add_argument("--layer-index", type=int, default=0,
                    help="array index inside hidden_c.npy, not the model layer number")
    ap.add_argument("--source-probe", default="out/main_qwen08_q4/probe_model.npz")
    ap.add_argument("--sizes", default="50,100,150,200,300")
    ap.add_argument("--lambdas", default="30,300,1000")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/main_qwen08_q4_hotpotqa_v2/refit_curve.json")
    args = ap.parse_args()

    t0 = time.time()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    lambdas = [float(s) for s in args.lambdas.split(",") if s.strip()]

    X, y, lp, ne, hidden, scores_path = load_rows(
        args.run, args.scores, args.label_column, args.layer_index)
    if len(y) < max(sizes) + args.cv_folds:
        print("not enough rows", file=sys.stderr)
        return 1
    print("rows=%d pos=%d base=%.3f" % (len(y), int(y.sum()), float(y.mean())))

    z = np.load(args.source_probe)
    w_src = z["w"].astype(np.float64)
    mu_src = z["mu"].astype(np.float64)
    sd_src = z["sd"].astype(np.float64)
    if w_src.size - 1 != X.shape[1]:
        print("source probe width %d does not match feature width %d"
              % (w_src.size - 1, X.shape[1]), file=sys.stderr)
        return 1

    curve = {}
    for n_train in sizes:
        acc = {k: [] for k in ("scratch", "finetune", "frozen", "logprob", "entropy")}
        lams = {"scratch": [], "finetune": []}
        train_pos, test_n = [], []
        for rep in range(args.repeats):
            rng = np.random.default_rng(1000 + rep)
            tr, te = stratified_split(y, n_train, rng)
            train_pos.append(int(y[tr].sum()))
            test_n.append(int(len(te)))

            a, lam = fit_and_eval(X, y, tr, te, rng, lambdas, args.val_frac)
            acc["scratch"].append(a)
            lams["scratch"].append(lam)

            a, lam = fit_and_eval(X, y, tr, te, rng, lambdas, args.val_frac,
                                  mu=mu_src, sd=sd_src, w0=w_src)
            acc["finetune"].append(a)
            lams["finetune"].append(lam)

            acc["frozen"].append(auroc(y[te].astype(np.float64),
                                       score(w_src, X[te], mu_src, sd_src)))
            acc["logprob"].append(auroc(y[te].astype(np.float64), lp[te]))
            acc["entropy"].append(auroc(y[te].astype(np.float64), ne[te]))

        row = {}
        for k, v in acc.items():
            a = np.array(v, dtype=np.float64)
            row[k] = {"auroc_mean": float(np.nanmean(a)),
                      "auroc_sd": float(np.nanstd(a, ddof=1))}
        row["train_pos_mean"] = float(np.mean(train_pos))
        row["test_rows_mean"] = float(np.mean(test_n))
        row["lambda_median"] = {k: float(np.median(v)) for k, v in lams.items()}
        curve[str(n_train)] = row
        print("n=%3d pos~%3.0f test~%3.0f | scratch %.3f+-%.3f | finetune %.3f+-%.3f | "
              "frozen %.3f | logprob %.3f | entropy %.3f"
              % (n_train, row["train_pos_mean"], row["test_rows_mean"],
                 row["scratch"]["auroc_mean"], row["scratch"]["auroc_sd"],
                 row["finetune"]["auroc_mean"], row["finetune"]["auroc_sd"],
                 row["frozen"]["auroc_mean"], row["logprob"]["auroc_mean"],
                 row["entropy"]["auroc_mean"]), flush=True)

    paired = {}
    for n_train in sizes:
        d_ent, d_lp = [], []
        for rep in range(args.repeats):
            rng = np.random.default_rng(1000 + rep)
            tr, te = stratified_split(y, n_train, rng)
            a, _ = fit_and_eval(X, y, tr, te, rng, lambdas, args.val_frac)
            d_ent.append(a - auroc(y[te].astype(np.float64), ne[te]))
            d_lp.append(a - auroc(y[te].astype(np.float64), lp[te]))
        d_ent = np.array(d_ent)
        d_lp = np.array(d_lp)
        paired[str(n_train)] = {
            "scratch_minus_entropy": {"mean": float(d_ent.mean()),
                                      "sd": float(d_ent.std(ddof=1))},
            "scratch_minus_logprob": {"mean": float(d_lp.mean()),
                                      "sd": float(d_lp.std(ddof=1))},
        }

    cv = kfold_cv(X, y, args.cv_folds, lambdas, args.val_frac, args.seed)
    print("cv%d %.3f+-%.3f" % (args.cv_folds, cv["auroc_mean"], cv["auroc_sd"]))

    res = {
        "kind": "probe_refit_learning_curve",
        "run": args.run.replace("\\", "/"),
        "feature": {"point": "C", "array_index": args.layer_index,
                    "width": int(X.shape[1]), "dtype": "float64 (loaded from float16)"},
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "label_column": args.label_column,
        "split_scheme": ("stratified random split per repeat; lambda chosen on a "
                         "%.0f%% val slice of train; test scored once" % (100 * args.val_frac)),
        "sizes": sizes,
        "lambdas": lambdas,
        "repeats": args.repeats,
        "val_frac": args.val_frac,
        "curve": curve,
        "paired": paired,
        "cv": cv,
        "seconds": round(time.time() - t0, 1),
        "provenance": {
            "hidden": hidden.replace("\\", "/"),
            "hidden_sha256": sha256(hidden),
            "scores": scores_path.replace("\\", "/"),
            "scores_sha256": sha256(scores_path),
            "source_probe": args.source_probe.replace("\\", "/"),
            "source_probe_sha256": sha256(args.source_probe),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
