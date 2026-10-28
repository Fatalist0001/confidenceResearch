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

    npz = np.load(args.model, allow_pickle=True)
    w = np.asarray(npz["w"], dtype=np.float64)
    mu = np.asarray(npz["mu"], dtype=np.float64)
    sd = np.asarray(npz["sd"], dtype=np.float64)
    info = json.loads(str(npz["meta"])) if "meta" in npz else {}
    point = info.get("point", "C")
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
    if arr.shape[2] != mu.shape[0]:
        raise SystemExit(f"target D={arr.shape[2]} != probe D={mu.shape[0]}")
    if arr.shape[1] <= layer:
        raise SystemExit(f"target has {arr.shape[1]} layers, probe needs index {layer}")

    X = np.asarray(arr[idx, layer, :], dtype=np.float64)
    z = (X - mu) / sd
    score = w[0] + z @ w[1:]

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
