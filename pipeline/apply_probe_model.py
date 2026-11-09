"""Apply a saved probe (probe_model.npz) to another run's hidden states.

Used for the quant check: the probe selected on Qwen 0.8B BF16 applied as-is to the
Q4_K_M states of the same model, scored against the Q4 run's own labels.

    python apply_probe_model.py --model out/main_qwen08/probe_model.npz \
        --dir out/main_qwen08_q4 [--split test] [--out result.json]
"""
import argparse
import csv
import json
import os

import numpy as np

from baseline import auroc, auprc


def load_probe_model(path):
    """Load portable weights without pickle; never fit or change standardization."""
    with np.load(path, allow_pickle=False) as saved:
        info = json.loads(str(saved["meta"]))
        w = np.asarray(saved["w"], dtype=np.float64)
        mu = np.asarray(saved["mu"], dtype=np.float64)
        sd = np.asarray(saved["sd"], dtype=np.float64)
    if info.get("point") not in ("A", "C"):
        raise ValueError("saved probe must specify point A or C")
    if w.ndim != 1 or mu.ndim != 1 or sd.shape != mu.shape or w.shape != (len(mu) + 1,):
        raise ValueError("invalid saved w/mu/sd shapes")
    if not all(np.isfinite(x).all() for x in (w, mu, sd)) or not (sd > 0).all():
        raise ValueError("saved weights must be finite and every train sd must be positive")
    return {"info": info, "w": w, "mu": mu, "sd": sd}


def score_hidden(probe, hidden, rows, layers):
    """Resolve the frozen model layer ID in the target, not the source array index."""
    info = probe["info"]
    layer_id = int(info["layer_id"] if "layer_id" in info else info["layer"])
    if hidden.ndim != 3 or hidden.shape[1] != len(layers) or len(set(layers)) != len(layers):
        raise ValueError("target hidden array does not match its layer list")
    if layer_id not in layers:
        raise ValueError(f"target has no frozen model layer {layer_id}")
    if hidden.shape[2] != len(probe["mu"]):
        raise ValueError(f"target D={hidden.shape[2]} != probe D={len(probe['mu'])}")
    layer = layers.index(layer_id)
    idx = np.asarray(rows, dtype=np.int64)
    if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= hidden.shape[0]):
        raise ValueError("target row index is out of range")
    x = np.asarray(hidden[idx, layer, :], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite target hidden states")
    z = (x - probe["mu"]) / probe["sd"]
    return probe["w"][0] + z @ probe["w"][1:], layer


def read_labels(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["row"])] = r
    return out


def usable_rows(meta_path, labels, split):
    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") != split:
                continue
            lab = labels.get(int(rec["row"]))
            if lab is None:
                continue
            if rec.get("status") != "ok" or not rec.get("generated") or rec.get("truncated"):
                continue
            v = lab["correct_filtered_b"].strip()
            if v == "":
                continue
            rows.append((int(rec["row"]), int(v)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="probe_model.npz of the source run")
    ap.add_argument("--dir", required=True, help="target run directory with meta/labels/hidden_*.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out and os.path.exists(args.out):
        raise SystemExit(f"output already exists: {args.out}; choose a new filename")
    probe = load_probe_model(args.model)
    info = probe["info"]
    point = info["point"]
    layer = int(info.get("layer", 0))
    layer_id = int(info.get("layer_id", layer))
    print(f"probe: point {point}, array layer {layer} (model layer {layer_id}), "
          f"lambda {info.get('lambda')}, source {info.get('run', '?')}")

    labels = read_labels(os.path.join(args.dir, "labels.csv"))
    rows = usable_rows(os.path.join(args.dir, "meta.jsonl"), labels, args.split)
    idx = np.array([r for r, _ in rows], dtype=np.int64)
    y = np.array([v for _, v in rows], dtype=float)
    print(f"{args.split}: {len(rows)} rows, {int(y.sum())} positives")

    arr = np.load(os.path.join(args.dir, f"hidden_{point.lower()}.npy"), mmap_mode="r")
    with open(os.path.join(args.dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    score, layer = score_hidden(probe, arr, idx, [int(x) for x in config["layers"]])

    res = {
        "model": args.model,
        "dir": args.dir,
        "split": args.split,
        "point": point,
        "layer": layer,
        "layer_id": layer_id,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "auroc": auroc(y, score),
        "auprc": auprc(y, score),
        "score_mean": float(score.mean()),
        "score_sd": float(score.std()),
    }
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
                     ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
