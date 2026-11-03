"""Shared builders for the unit tests.

Everything here writes only into a temporary directory. The tests never touch
pipeline/out and never need a model or a GPU.
"""

import csv
import json
import os
import sys

PIPELINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PIPELINE)
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)


def _np():
    import numpy as np  # imported lazily so the failure message is clear
    return np


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_run(path, n_per_split=30, layers=(3, 7, 11), n_embd=5, n_layers=12,
             signal=True, seed=0, labels=True):
    """Write a complete, self-consistent run directory.

    With signal=True the label is recoverable from layer index 1, so a probe must
    find it. With signal=False the labels are noise and the probe must not.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    os.makedirs(path, exist_ok=True)

    names = ("train", "val", "test")
    rows, label_rows = [], []
    total = n_per_split * len(names)
    a = np.zeros((total, len(layers), n_embd), dtype=np.float16)
    c = np.zeros_like(a)

    for split_i, split in enumerate(names):
        for j in range(n_per_split):
            row = split_i * n_per_split + j
            y = int(rng.integers(0, 2))
            rows.append({
                "row": row,
                "id": f"{split}_{j}",
                "split": split,
                "status": "ok",
                "generated": True,
                "truncated": False,
                "pos_a": 10,
                "pos_c": 12,
                "n_tokens": 13,
                "answer": f"answer {y}",
                "gold": [f"answer {y}"],
            })
            label_rows.append({
                "id": f"{split}_{j}",
                "row": row,
                "split": split,
                "correct_filtered_b": y,
            })
            vec_a = rng.normal(size=(len(layers), n_embd)).astype(np.float16)
            vec_c = rng.normal(size=(len(layers), n_embd)).astype(np.float16)
            if signal:
                # layer 1 carries the label in its first coordinate
                vec_a[1, 0] = (1.8 if y else -1.8) + 0.3 * rng.normal()
                vec_c[1, 0] = (1.8 if y else -1.8) + 0.3 * rng.normal()
            a[row] = vec_a
            c[row] = vec_c

    write_json(os.path.join(path, "config.json"), {
        "model": "synthetic.gguf",
        "model_name": "synthetic",
        "n_layers": n_layers,
        "n_layers_selected": len(layers),
        "layers": list(layers),
        "n_embd": n_embd,
        "n_examples": total,
        "generate": True,
        "max_len": 1024,
        "max_new_tokens": 64,
        "template": True,
        "thinking": False,
        "batch": "data/synthetic.jsonl",
        "llama_cpp_commit": "785322d17",
        "llama_cpp_build": 12033,
    })
    write_jsonl(os.path.join(path, "meta.jsonl"), rows)

    np.save(os.path.join(path, "hidden_a.npy"), a)
    np.save(os.path.join(path, "hidden_c.npy"), c)

    if labels:
        fields = ["id", "row", "split", "correct_filtered_b"]
        with open(os.path.join(path, "labels.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(label_rows)
    return path


def make_input(path, n=200, instruction="Answer with just the answer, no explanation.",
               duplicate_instruction=False, with_splits=False):
    rows = []
    for i in range(n):
        text = f"Question number {i}?\n{instruction}"
        if duplicate_instruction and i == 3:
            text += "\n" + instruction
        row = {"id": f"q_{i}", "question": text}
        if with_splits:
            block = max(1, n // 3)
            row["split"] = ("train", "val", "test")[min(i // block, 2)]
        rows.append(row)
    write_jsonl(path, rows)
    return path


def write_log(path, rows, ok=None, skipped=0, truncated=0, seconds=40.0, expected=None):
    ok = rows if ok is None else ok
    expected = rows if expected is None else expected
    per_s = rows / seconds if seconds else 0.0
    with open(path, "w", encoding="utf-8") as f:
        f.write("some startup noise\n")
        f.write(f"done: {rows}/{expected}  ok={ok} skipped={skipped} truncated={truncated}"
                f"  in {seconds:.1f} s ({per_s:.2f} ex/s)\n")
        f.write("retokenize mismatch: 0/0\n")
    return path
