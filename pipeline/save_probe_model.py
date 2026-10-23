"""Rebuild the selected probe of an existing run and save its weights.

probe.py stores only metrics in probe.json; the fitted model itself is
deterministic, so it can be rebuilt from the same train split, standardization
and lambda. The result is written next to probe.json as probe_model.npz
(arrays w, mu, sd and a JSON `meta`).

Usage:
    python save_probe_model.py --dir out/main_gemma4e2b
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from probe import (
    auroc,
    fit_l2,
    read_jsonl,
    read_labels,
    save_probe_model,
    split_indices,
    standardize,
    with_bias,
)


def read_scores(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append((r["split"], int(r["row"]), float(r["score"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run dir with probe.json, meta.jsonl, labels.csv, hidden_*.npy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.dir
    with open(os.path.join(d, "probe.json"), encoding="utf-8") as f:
        pj = json.load(f)
    sel = pj["selected"]
    lam = float(pj["selected_lambda"])

    meta = read_jsonl(os.path.join(d, "meta.jsonl"))
    labels = read_labels(os.path.join(d, "labels.csv"))
    idx = split_indices(meta, labels)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    ytr = np.array([int(labels[meta[i]["row"]]["correct_filtered_b"]) for i in tr], dtype=float)
    yva = np.array([int(labels[meta[i]["row"]]["correct_filtered_b"]) for i in va], dtype=float)
    yte = np.array([int(labels[meta[i]["row"]]["correct_filtered_b"]) for i in te], dtype=float)

    hidden = os.path.join(d, "hidden_%s.npy" % sel["point"].lower())
    arr = np.load(hidden, mmap_mode="r")
    Xtr = np.asarray(arr[tr, sel["layer"], :], dtype=np.float64)
    Xva = np.asarray(arr[va, sel["layer"], :], dtype=np.float64)
    Xte = np.asarray(arr[te, sel["layer"], :], dtype=np.float64)
    Ztr, (Zva, Zte) = standardize(Xtr, [Xva, Xte])
    w = fit_l2(Ztr, ytr, lam)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8

    s_te = with_bias(Zte) @ w
    got = auroc(yte, s_te)
    want = pj["metrics"]["test"]["auroc"]
    print("%s: point %s layer %d lambda %g | test AUROC rebuilt %.9f, probe.json %.9f, diff %.3g"
          % (d, sel["point"], sel["layer"], lam, got, want, abs(got - want)))
    if abs(got - want) > 1e-9:
        print("error: rebuilt probe does not reproduce probe.json", file=sys.stderr)
        return 1

    scores_path = os.path.join(d, "probe_scores.csv")
    if os.path.exists(scores_path):
        ref = {row: sc for split, row, sc in read_scores(scores_path) if split == "test"}
        diff = max(abs(ref[meta[i]["row"]] - s_te[j]) for j, i in enumerate(te))
        print("  max |score - probe_scores.csv| on test: %.3g" % diff)

    out = args.out or os.path.join(d, "probe_model.npz")
    save_probe_model(
        out, w, mu, sd,
        {
            "version": 1,
            "kind": "l2_logistic_probe",
            "point": sel["point"],
            "layer": int(sel["layer"]),
            "lambda": lam,
            "label_column": "correct_filtered_b",
            "w_layout": "[intercept, coef[D]]",
            "apply": "z = (x - mu) / sd; score = w[0] + w[1:] @ z",
            "standardization": "mean/std over the train split",
            "hidden": hidden.replace("\\", "/"),
            "meta": os.path.join(d, "meta.jsonl").replace("\\", "/"),
            "labels": os.path.join(d, "labels.csv").replace("\\", "/"),
            "rows": {"train": len(tr), "val": len(va), "test": len(te)},
            "positives": {"train": int(ytr.sum()), "val": int(yva.sum()), "test": int(yte.sum())},
        },
    )
    print("  saved %s (%.1f KB)" % (out, os.path.getsize(out) / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
