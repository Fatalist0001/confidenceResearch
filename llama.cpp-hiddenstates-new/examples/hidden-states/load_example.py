#!/usr/bin/env python3
"""Inspect a batch output produced by `llama-hidden-states --batch`.

Usage:
    python load_example.py <out_dir> [--k 3]

Reads config.json, hidden_a.npy / hidden_c.npy via np.load(mmap_mode="r")
and meta.jsonl, then prints shapes plus a few metadata rows.
"""

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", help="output directory (the --out value of the batch run)")
    ap.add_argument("--k", type=int, default=3, help="how many meta.jsonl rows to print")
    args = ap.parse_args()

    out_dir = args.dir

    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        print("config.json:")
        for key in (
            "model",
            "n_layers",
            "n_embd",
            "layers",
            "n_layers_selected",
            "max_len",
            "max_new_tokens",
            "template",
            "thinking",
            "generate",
            "n_examples",
            "llama_cpp_build",
        ):
            if key in cfg:
                print(f"  {key}: {cfg[key]}")
        defs = cfg.get("point_definitions")
        if defs:
            print("  point_definitions:")
            for name, text in defs.items():
                print(f"    {name}: {text}")
    else:
        print("config.json: missing")

    arrays = {}
    for name in ("hidden_c.npy", "hidden_a.npy"):
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            print(f"{name}: missing")
            continue
        arr = np.load(path, mmap_mode="r")
        arrays[name] = arr
        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
        if arr.size:
            finite = np.isfinite(np.asarray(arr[0], dtype=np.float32)).all()
            print(f"  first row finite: {finite}")

    meta_path = os.path.join(out_dir, "meta.jsonl")
    if not os.path.exists(meta_path):
        print("meta.jsonl: missing")
        return

    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"meta.jsonl: {len(rows)} rows ({n_ok} ok, {len(rows) - n_ok} skipped)")

    for r in rows[: args.k]:
        ans = r.get("answer", "") or ""
        shown = ans if len(ans) <= 60 else ans[:60] + "..."
        print(
            "  {id!r} row={row} n_tokens={n_tokens} pos_a={pos_a} pos_c={pos_c} "
            "generated={generated} truncated={truncated} status={status} answer={ans!r}".format(
                id=r.get("id"),
                row=r.get("row"),
                n_tokens=r.get("n_tokens"),
                pos_a=r.get("pos_a"),
                pos_c=r.get("pos_c"),
                generated=r.get("generated"),
                truncated=r.get("truncated"),
                status=r.get("status"),
                ans=shown,
            )
        )
        if r.get("reason"):
            print(f"      reason: {r['reason']}")

    n_trunc = sum(1 for r in rows if r.get("truncated"))
    if n_trunc:
        print(f"  truncated: {n_trunc}/{len(rows)}")

    # cross-check meta vs arrays
    if arrays:
        n_rows_meta = len(rows)
        for name, arr in arrays.items():
            if arr.shape[0] != n_rows_meta and n_rows_meta:
                print(f"  warning: {name} has {arr.shape[0]} rows but meta.jsonl has {n_rows_meta}")


if __name__ == "__main__":
    main()
