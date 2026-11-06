"""Build and evaluate the fixed-budget LFM2.5 Thinking protocol.

The protocol is deliberately split into three passes:

1. generate at most R reasoning tokens with the stock chat template;
2. append ``</think>\nFinal answer:`` and generate a final continuation;
3. extract the first completed final line and run hidden-state extraction with that
   line supplied as the answer.

Budget selection uses a train-only development set.  Test labels are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


INSTRUCTION = "Answer with just the answer, no explanation."
REASONING_RESTART_RE = re.compile(r"\s+(?:wait|however|alternatively|but|hmm)\b", re.IGNORECASE)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def align_rows(left: list[dict], right: list[dict], left_name: str, right_name: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"row count differs: {left_name}={len(left)}, {right_name}={len(right)}")
    for i, (a, b) in enumerate(zip(left, right)):
        if a.get("id") != b.get("id"):
            raise ValueError(
                f"id differs at row {i}: {left_name}={a.get('id')!r}, {right_name}={b.get('id')!r}"
            )


def read_meta_rows(path: str | Path) -> list[dict]:
    """Read meta.jsonl, tolerating one truncated final line left by a killed run."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"invalid JSON at line {index + 1} of {path}")
    return rows


def meta_count(path: str | Path) -> dict:
    rows = read_meta_rows(path)
    return {
        "rows": len(rows),
        "last_id": rows[-1]["id"] if rows else None,
        "last_row": rows[-1].get("row") if rows else None,
    }


def make_remaining(source: str | Path, out: str | Path, start: int) -> dict:
    rows = read_jsonl(source)
    if not 0 <= start < len(rows):
        raise ValueError(f"start must be in [0, {len(rows)}), got {start}")
    remaining = rows[start:]
    write_jsonl(out, remaining)
    return {
        "rows": len(remaining),
        "start": start,
        "first_id": remaining[0]["id"],
        "last_id": remaining[-1]["id"],
    }


def merge_reasoning(
    questions_path: str | Path,
    part_paths: list[str | Path],
    out_path: str | Path,
) -> dict:
    """Concatenate contiguous meta parts and restore global row indices."""
    questions = read_jsonl(questions_path)
    merged: list[dict] = []
    parts = []
    for part_path in part_paths:
        rows = read_meta_rows(part_path)
        for local, row in enumerate(rows):
            if row.get("row") != local:
                raise ValueError(f"{part_path}: row {local} has row={row.get('row')!r}")
        parts.append({"path": str(part_path), "rows": len(rows)})
        merged.extend(rows)
    if len(merged) != len(questions):
        raise ValueError(f"merged rows {len(merged)} != questions {len(questions)}")
    for index, (row, question) in enumerate(zip(merged, questions)):
        if row.get("id") != question.get("id"):
            raise ValueError(
                f"id differs at row {index}: meta={row.get('id')!r}, questions={question.get('id')!r}"
            )
        if row.get("status") != "ok" or row.get("generated") is not True:
            raise ValueError(f"reasoning row {index} is not generated: status={row.get('status')!r}")
        row["row"] = index
    write_jsonl(out_path, merged)
    return {"rows": len(merged), "parts": parts, "out": str(out_path)}


def make_dev(source: str | Path, out: str | Path, split: str, count: int) -> dict:
    rows = [row for row in read_jsonl(source) if row.get("split") == split][:count]
    if len(rows) != count:
        raise ValueError(f"wanted {count} rows from split {split!r}, found {len(rows)}")
    write_jsonl(out, rows)
    return {"rows": len(rows), "split": split, "first_id": rows[0]["id"], "last_id": rows[-1]["id"]}


def render_lfm_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def reasoning_token_count(row: dict) -> int | None:
    pos_a = row.get("pos_a")
    pos_c = row.get("pos_c")
    if isinstance(pos_a, int) and isinstance(pos_c, int) and pos_c >= pos_a:
        return pos_c - pos_a
    return None


def build_final_prompts(
    questions_path: str | Path,
    reasoning_meta_path: str | Path,
    out_path: str | Path,
    budget: int,
) -> dict:
    questions = read_jsonl(questions_path)
    reasoning_rows = read_jsonl(reasoning_meta_path)
    align_rows(questions, reasoning_rows, "questions", "reasoning")

    built = []
    natural_closes = 0
    source_truncated = 0
    for question_row, meta in zip(questions, reasoning_rows):
        original_question = question_row.get("question") or ""
        if original_question.count(INSTRUCTION) != 1:
            raise ValueError(f"source instruction count is not one for id {question_row.get('id')}")
        if meta.get("status") != "ok" or not meta.get("generated"):
            raise ValueError(f"reasoning generation failed for id {question_row.get('id')}")
        reasoning = meta.get("answer") or ""
        natural_close = "</think>" in reasoning
        if natural_close:
            reasoning = reasoning.split("</think>", 1)[0]
            natural_closes += 1
        if not reasoning.lstrip().startswith("<think>"):
            reasoning = "<think>\n" + reasoning
        reasoning = reasoning.rstrip()
        prompt = render_lfm_prompt(original_question)
        prompt += reasoning + "\n</think>\nFinal answer:"

        row = dict(question_row)
        row["original_question"] = original_question
        row["question"] = prompt
        row["reasoning"] = reasoning
        row["reasoning_budget"] = budget
        row["reasoning_n_tokens"] = reasoning_token_count(meta)
        row["reasoning_closed_naturally"] = natural_close
        row["reasoning_source_truncated"] = bool(meta.get("truncated"))
        source_truncated += int(bool(meta.get("truncated")))
        built.append(row)

    write_jsonl(out_path, built)
    return {
        "rows": len(built),
        "budget": budget,
        "natural_closes": natural_closes,
        "reasoning_source_truncated": source_truncated,
    }


def extract_first_final_line(raw: str, generation_truncated: bool) -> tuple[str, str]:
    """Return (answer, reason).  A line is complete only at newline or natural EOG."""
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        content = line.rstrip("\r\n").rstrip()
        if not content.strip():
            continue
        marker_positions = [pos for marker in ("<think", "</think", "<|") if (pos := content.find(marker)) >= 0]
        marker_stop = bool(marker_positions)
        if marker_stop:
            content = content[: min(marker_positions)].rstrip()
            if not content.strip():
                return "", "reasoning_reopened"
        restart = REASONING_RESTART_RE.search(content)
        restart_stop = bool(restart and content[: restart.start()].strip())
        if restart_stop:
            content = content[: restart.start()].rstrip()
        ended_by_newline = line.endswith(("\n", "\r"))
        is_last = index == len(lines) - 1
        ended_by_eog = is_last and not generation_truncated
        if not (marker_stop or restart_stop or ended_by_newline or ended_by_eog):
            return "", "no_final_stop"
        if marker_stop:
            return content, "marker"
        if restart_stop:
            return content, "reasoning_restart"
        return content, "newline" if ended_by_newline else "eog"
    return "", "empty"


def extract_finals(
    prompts_path: str | Path,
    final_meta_path: str | Path,
    state_input_path: str | Path,
    eval_meta_path: str | Path,
    summary_path: str | Path,
) -> dict:
    prompts = read_jsonl(prompts_path)
    final_rows = read_jsonl(final_meta_path)
    align_rows(prompts, final_rows, "prompts", "final")

    state_rows = []
    eval_rows = []
    reasons: Counter[str] = Counter()
    for i, (prompt, final_meta) in enumerate(zip(prompts, final_rows)):
        raw = final_meta.get("answer") or ""
        answer, reason = extract_first_final_line(raw, bool(final_meta.get("truncated")))
        reasons[reason] += 1
        success = bool(answer)

        state_row = dict(prompt)
        state_row["answer"] = answer
        state_row["final_raw"] = raw
        state_row["final_extraction"] = reason
        state_row["final_generated"] = True
        state_row["final_generation_truncated"] = bool(final_meta.get("truncated"))
        state_row["final_generation_n_tokens"] = reasoning_token_count(final_meta)
        state_row["final_logprobs"] = final_meta.get("logprobs")
        state_row["final_entropies"] = final_meta.get("entropies")
        state_rows.append(state_row)

        eval_row = dict(prompt)
        eval_row.update(
            {
                "row": i,
                "status": "ok",
                "generated": True,
                "truncated": not success,
                "answer": answer,
                "final_raw": raw,
                "final_extraction": reason,
            }
        )
        eval_rows.append(eval_row)

    summary = {
        "rows": len(prompts),
        "extracted": len(prompts)
        - sum(v for k, v in reasons.items() if k not in ("newline", "eog", "marker", "reasoning_restart")),
        "coverage": 0.0,
        "reasons": dict(sorted(reasons.items())),
    }
    summary["coverage"] = summary["extracted"] / summary["rows"] if summary["rows"] else 0.0
    write_jsonl(state_input_path, state_rows)
    write_jsonl(eval_meta_path, eval_rows)
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def finalize_replay(replay_meta_path: str | Path, out_path: str | Path) -> dict:
    rows = read_jsonl(replay_meta_path)
    extraction_failures = 0
    for row in rows:
        logprobs = row.get("final_logprobs")
        entropies = row.get("final_entropies")
        if row.get("final_generated") is not True:
            raise ValueError(f"final_generated is not true for id {row.get('id')}")
        if not isinstance(logprobs, list) or not logprobs:
            raise ValueError(f"final logprobs are missing for id {row.get('id')}")
        if not isinstance(entropies, list) or not entropies:
            raise ValueError(f"final entropies are missing for id {row.get('id')}")
        failed = not bool(str(row.get("answer") or "").strip())
        extraction_failures += int(failed)
        row["generated"] = True
        row["truncated"] = False
        row["logprobs"] = logprobs
        row["entropies"] = entropies
        row.pop("final_logprobs", None)
        row.pop("final_entropies", None)
        row.pop("final_generated", None)
        row["final_extraction_failed"] = failed
        row["generation_protocol"] = "lfm25-thinking-final-v1"
    write_jsonl(out_path, rows)
    return {"rows": len(rows), "extraction_failures": extraction_failures}


def parse_candidate(value: str) -> tuple[int, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or "," not in parts[1]:
        raise argparse.ArgumentTypeError("candidate must be BUDGET=labels.csv,summary.json")
    labels, summary = parts[1].split(",", 1)
    try:
        budget = int(parts[0])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate budget must be an integer") from exc
    return budget, Path(labels), Path(summary)


def candidate_metrics(budget: int, labels_path: Path, summary_path: Path) -> dict:
    with open(labels_path, encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    total = len(labels)
    labeled = [row for row in labels if row.get("correct_filtered_b") in ("0", "1")]
    positives = sum(row["correct_filtered_b"] == "1" for row in labeled)
    return {
        "budget": budget,
        "rows": total,
        "extracted": int(summary["extracted"]),
        "coverage": float(summary["coverage"]),
        "labeled": len(labeled),
        "positives": positives,
        "labeled_accuracy": positives / len(labeled) if labeled else 0.0,
        "effective_accuracy": positives / total if total else 0.0,
        "labels": str(labels_path),
        "summary": str(summary_path),
    }


def select_budget(
    candidates: list[tuple[int, Path, Path]],
    out_path: str | Path,
    min_coverage: float,
    margin: float,
) -> dict:
    metrics = [candidate_metrics(*candidate) for candidate in candidates]
    metrics.sort(key=lambda row: row["budget"])
    eligible = [row for row in metrics if row["coverage"] >= min_coverage]
    if not eligible:
        raise ValueError(f"no candidate reached coverage >= {min_coverage:.3f}")
    best_accuracy = max(row["effective_accuracy"] for row in eligible)
    selected = next(row for row in eligible if row["effective_accuracy"] >= best_accuracy - margin)
    result = {
        "selection_data": "train-only development rows",
        "criterion": (
            f"smallest budget with coverage >= {min_coverage:.3f} and effective accuracy "
            f"within {margin:.3f} of the best eligible candidate"
        ),
        "min_coverage": min_coverage,
        "margin": margin,
        "best_effective_accuracy": best_accuracy,
        "selected_budget": selected["budget"],
        "candidates": metrics,
    }
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-dev", help="select a deterministic split-only development set")
    p.add_argument("--in", dest="source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--rows", type=int, default=200)

    p = sub.add_parser("make-remaining", help="write source rows from a global start index")
    p.add_argument("--in", dest="source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, required=True)

    p = sub.add_parser("meta-count", help="count complete meta rows and tolerate a cut tail")
    p.add_argument("--meta", required=True)

    p = sub.add_parser("merge-reasoning", help="merge contiguous reasoning meta parts in order")
    p.add_argument("--questions", required=True)
    p.add_argument("--part", action="append", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("build-final-prompts", help="append a forced final cue after reasoning")
    p.add_argument("--questions", required=True)
    p.add_argument("--reasoning-meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, required=True)

    p = sub.add_parser("extract-finals", help="extract first completed final lines")
    p.add_argument("--prompts", required=True)
    p.add_argument("--final-meta", required=True)
    p.add_argument("--state-input", required=True)
    p.add_argument("--eval-meta", required=True)
    p.add_argument("--summary", required=True)

    p = sub.add_parser("finalize-replay", help="restore final-generation provenance after state replay")
    p.add_argument("--replay-meta", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("select-budget", help="select the smallest near-best train-only budget")
    p.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "make-dev":
        result = make_dev(args.source, args.out, args.split, args.rows)
    elif args.command == "make-remaining":
        result = make_remaining(args.source, args.out, args.start)
    elif args.command == "meta-count":
        result = meta_count(args.meta)
    elif args.command == "merge-reasoning":
        result = merge_reasoning(args.questions, args.part, args.out)
    elif args.command == "build-final-prompts":
        result = build_final_prompts(args.questions, args.reasoning_meta, args.out, args.budget)
    elif args.command == "extract-finals":
        result = extract_finals(args.prompts, args.final_meta, args.state_input, args.eval_meta, args.summary)
    elif args.command == "finalize-replay":
        result = finalize_replay(args.replay_meta, args.out)
    else:
        result = select_budget(args.candidate, args.out, args.min_coverage, args.margin)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
