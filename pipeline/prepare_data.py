"""Prepare TriviaQA (rc.nocontext, validation) for the confidence experiment.

Loads the dataset from Hugging Face, deduplicates and filters questions,
assigns each question to train/val/test by its answer group (deterministic
hash of group + seed), samples --n questions preserving split proportions,
and writes one JSONL row per question plus a meta JSON file.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

from datasets import load_dataset

from normalize import normalize

DATASET_NAME = "mandarjoshi/trivia_qa"
DATASET_CONFIG = "rc.nocontext"
DATASET_SPLIT = "validation"
DATASET_TAG = "triviaqa-rc-nocontext-validation"

QUESTION_TEMPLATE = "{question}\nAnswer with just the answer, no explanation."

MAX_QUESTION_CHARS = 300
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def assign_split(group: str, seed: int) -> str:
    """Deterministically map a group to train/val/test by hash of group+seed."""
    h = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    bucket = int.from_bytes(h[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000, help="number of questions to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--out", type=str, default="data/data.jsonl", help="output JSONL path")
    ap.add_argument("--meta", type=str, default="data/meta.json", help="output meta JSON path")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG}), split={DATASET_SPLIT} ...", flush=True)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    print(f"  raw rows: {len(ds)}", flush=True)

    # Keep only the fields we need.
    kept = []
    for ex in ds:
        kept.append(
            {
                "question_id": ex["question_id"],
                "question": ex["question"],
                "answer_value": ex["answer"]["value"],
                "answer_aliases": list(ex["answer"]["aliases"] or []),
            }
        )

    # Dedup by normalize(question); keep first.
    seen = {}
    deduped = []
    n_dupes = 0
    for row in kept:
        key = normalize(row["question"])
        if key in seen:
            n_dupes += 1
            continue
        seen[key] = True
        deduped.append(row)

    # Drop too-long questions and questions without a gold answer.
    filtered = []
    for row in deduped:
        q = row["question"]
        v = row["answer_value"]
        if len(q) > MAX_QUESTION_CHARS:
            continue
        if not v or not v.strip():
            continue
        filtered.append(row)

    # Group = normalize(answer.value). Assign split by hash of group+seed.
    for row in filtered:
        row["group"] = normalize(row["answer_value"])
        row["split"] = assign_split(row["group"], args.seed)

    # Sample --n questions, preserving split proportions (60/20/20).
    by_split = defaultdict(list)
    for row in filtered:
        by_split[row["split"]].append(row)

    total = len(filtered)
    quotas = {}
    acc = 0
    for i, sp in enumerate(("train", "val", "test")):
        if i < 2:
            q = round(args.n * SPLIT_RATIOS[sp])
            quotas[sp] = q
            acc += q
        else:
            quotas[sp] = args.n - acc

    sampled = []
    for sp in ("train", "val", "test"):
        pool = by_split[sp]
        rng.shuffle(pool)
        take = min(quotas[sp], len(pool))
        if take < len(pool):
            # keep original order within the split for reproducibility of ids
            pass
        sampled.extend(pool[:take])

    # Build output rows.
    out_rows = []
    for row in sampled:
        value = row["answer_value"]
        gold = [normalize(value)]
        for alias in row["answer_aliases"]:
            g = normalize(alias)
            if g and g not in gold:
                gold.append(g)
        out_rows.append(
            {
                "id": row["question_id"],
                "question": QUESTION_TEMPLATE.format(question=row["question"]),
                "gold": gold,
                "gold_main": value,
                "dataset": DATASET_TAG,
                "split": row["split"],
                "group": row["group"],
            }
        )

    # Ensure output dirs exist.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta) or ".", exist_ok=True)

    # Write JSONL.
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Meta.
    ids = [r["id"] for r in out_rows]
    ids_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    split_counts = defaultdict(int)
    for r in out_rows:
        split_counts[r["split"]] += 1
    n_groups = len({r["group"] for r in filtered})

    meta = {
        "seed": args.seed,
        "n": args.n,
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split_source": DATASET_SPLIT,
        "dataset_tag": DATASET_TAG,
        "question_template": QUESTION_TEMPLATE,
        "split_counts": {sp: split_counts[sp] for sp in ("train", "val", "test")},
        "n_groups": n_groups,
        "n_dupes_removed": n_dupes,
        "ids_sha256": ids_sha,
        "max_question_chars": MAX_QUESTION_CHARS,
        "total_after_filters": total,
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Console report.
    print("\n=== summary ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"n (requested): {args.n}", flush=True)
    print(f"dataset: {DATASET_NAME}  config: {DATASET_CONFIG}  split: {DATASET_SPLIT}", flush=True)
    print(f"question template: {QUESTION_TEMPLATE!r}", flush=True)
    print(f"rows kept after filters: {total}", flush=True)
    print(f"dupes removed: {n_dupes}", flush=True)
    print(f"groups: {n_groups}", flush=True)
    print(f"split counts (sampled): train={split_counts['train']}  val={split_counts['val']}  test={split_counts['test']}", flush=True)
    print(f"ids sha256: {ids_sha}", flush=True)
    print(f"wrote: {args.out}  ({len(out_rows)} rows)", flush=True)
    print(f"wrote: {args.meta}", flush=True)

    print("\n=== 5 random examples ===", flush=True)
    for r in rng.sample(out_rows, min(5, len(out_rows))):
        print(json.dumps(r, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
