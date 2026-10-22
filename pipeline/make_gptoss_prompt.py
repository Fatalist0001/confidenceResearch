"""Pre-render the gpt-oss harmony prompt with the `final` channel pre-filled.

`llama-hidden-states` ignores `--no-thinking` / `--reasoning*` for gpt-oss
(the chat template always asks for `Reasoning: medium`, and generation uses
raw greedy argmax without a sampler), so reasoning eats the token budget.
The workaround is to hand the tool a ready harmony prompt that stops right
before the assistant's `final` content and to run it with `--no-template`.

This script rewrites `question` in a JSONL copy of the dataset, keeps every
other key untouched, and strips the dataset's own answer instruction so it is
appended exactly once.

Example:
  python make_gptoss_prompt.py --in data/main.jsonl --out %TEMP%\\gptoss_final.jsonl
"""

import argparse
import json
import io

SYSTEM_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {date}\n"
    "\n"
    "Reasoning: medium\n"
    "\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)
PREFIX = "<|start|>system<|message|>{system}<|end|><|start|>user<|message|>"
INSTRUCTION = "\nAnswer with just the answer, no explanation."
SUFFIX = INSTRUCTION + "<|end|><|start|>assistant<|channel|>final<|message|>"


def strip_instruction(question: str) -> str:
    """Remove the dataset answer instruction so it is only appended once."""
    q = question.strip()
    if q.endswith(INSTRUCTION.strip()):
        q = q[: -len(INSTRUCTION.strip())].rstrip()
    for sep in ("\nAnswer with just the answer, no explanation.", "\nAnswer with just the answer"):
        if q.endswith(sep):
            q = q[: -len(sep)].rstrip()
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=str, default="data/main.jsonl",
                    help="input JSONL with `question` (default: data/main.jsonl)")
    ap.add_argument("--out", dest="dst", type=str, required=True, help="output JSONL path")
    ap.add_argument("--date", type=str, default="2026-10-21",
                    help="value for `Current date:` (default: 2026-10-21)")
    args = ap.parse_args()

    prefix = PREFIX.format(system=SYSTEM_TEMPLATE.format(date=args.date))

    rows = [json.loads(line) for line in io.open(args.src, encoding="utf-8") if line.strip()]
    with io.open(args.dst, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            row["question"] = prefix + strip_instruction(row["question"]) + SUFFIX
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dup = sum(1 for row in rows if row["question"].count("Answer with just the answer") != 1)
    print(f"rows: {len(rows)} | rows with duplicated instruction: {dup}")
    print(f"wrote: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
