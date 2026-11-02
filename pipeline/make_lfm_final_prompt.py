"""Pre-render an LFM2.5 chat prompt with reasoning closed and final answer cued.

LFM2.5-2.6B opens ``<think>`` even when thinking is disabled.  With the normal
template every 64-token pilot answer is truncated.  This script creates a JSONL
copy whose assistant prefix ends in ``<think></think>\nFinal answer:``.  Run the
copy with ``--no-template``; the tokenizer still adds the model BOS token.

Every key except ``question`` is preserved.  Re-running the script on its own
output is idempotent.
"""

import argparse
import io
import json


PREFIX = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think></think>\nFinal answer:"


def render(question: str) -> str:
    question = question.strip()
    if question.startswith(PREFIX) and question.endswith(SUFFIX):
        return question
    return PREFIX + question + SUFFIX


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="data/main.jsonl")
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in io.open(args.src, encoding="utf-8") if line.strip()]
    with io.open(args.dst, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            row["question"] = render(row["question"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bad = sum(row["question"].count("Answer with just the answer") != 1 for row in rows)
    print(f"rows: {len(rows)} | bad instruction count: {bad}")
    print(f"wrote: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
