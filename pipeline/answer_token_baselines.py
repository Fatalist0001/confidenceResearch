"""Exploratory answer-token confidence baselines for canonical TriviaQA runs.

The frozen primary baseline uses the first answer token.  This script adds
descriptive full-answer aggregates without changing any existing ``baseline.*``
artifact:

* mean, minimum, and last-token logprob;
* negative mean and negative last-token entropy.

It also recomputes the two primary first-token scores as integrity anchors.  Every
comparison uses the exact held-out test cohort of the saved probe, and paired
bootstrap resamples compare the probe with each token score.

The set of aggregates was inspected after the original test results were known.
Consequently these are supplementary exploratory comparisons, not a replacement
for the frozen primary protocol and not a preregistered choice of a new winner.

Usage (from the repository root):

    python pipeline/answer_token_baselines.py
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from baseline import auroc


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DATE = "2026-11-17"

METHODS = {
    "first_logprob": "first-token logprob",
    "first_neg_entropy": "negative first-token entropy",
    "mean_logprob": "mean answer-token logprob",
    "min_logprob": "minimum answer-token logprob",
    "last_logprob": "last answer-token logprob",
    "mean_neg_entropy": "negative mean answer-token entropy",
    "last_neg_entropy": "negative last-token entropy",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def source_info(path):
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def interval(values):
    return [float(x) for x in np.percentile(np.asarray(values), [2.5, 97.5])]


def aggregate_scores(logprobs, entropies, answer_tokens):
    """Return confidence scores over the first ``answer_tokens`` generation steps.

    Usually the score arrays end at point C.  The LFM2.5-Thinking replay preserves
    the complete fixed-size final-generation buffer after extracting the first
    answer line, so slicing by ``pos_c - pos_a`` is required there.
    """
    lp = np.asarray(logprobs[:answer_tokens], dtype=np.float64)
    h = np.asarray(entropies[:answer_tokens], dtype=np.float64)
    if answer_tokens <= 0 or len(lp) != answer_tokens or len(h) != answer_tokens:
        raise ValueError("token-score arrays do not cover the answer span")
    if not np.isfinite(lp).all() or not np.isfinite(h).all():
        raise ValueError("token-score arrays contain non-finite values")
    return {
        "first_logprob": float(lp[0]),
        "first_neg_entropy": float(-h[0]),
        "mean_logprob": float(lp.mean()),
        "min_logprob": float(lp.min()),
        "last_logprob": float(lp[-1]),
        "mean_neg_entropy": float(-h.mean()),
        "last_neg_entropy": float(-h[-1]),
    }


def summarize_run(root, name, entry, n_boot, seed):
    directory = root / "pipeline" / "out" / name
    source_paths = {
        key: directory / filename
        for key, filename in {
            "meta": "meta.jsonl",
            "labels": "labels.csv",
            "baseline": "baseline.json",
            "probe": "probe.json",
            "probe_scores": "probe_scores.csv",
        }.items()
    }
    for path in source_paths.values():
        if not path.exists():
            raise SystemExit(f"missing source artifact: {path}")

    label_rows = read_csv(source_paths["labels"])
    labels = {int(row["row"]): row for row in label_rows}
    if len(labels) != len(label_rows):
        raise SystemExit(f"duplicate label row in {source_paths['labels']}")

    probe_rows = [
        row for row in read_csv(source_paths["probe_scores"])
        if row["split"] == "test"
    ]
    probe = {int(row["row"]): row for row in probe_rows}
    if len(probe) != len(probe_rows):
        raise SystemExit(f"duplicate probe test row in {source_paths['probe_scores']}")

    ys = []
    probe_scores = []
    values = {key: [] for key in METHODS}
    lengths = []
    skipped = {
        "not_test": 0,
        "status": 0,
        "not_generated": 0,
        "truncated": 0,
        "no_label": 0,
        "no_scores": 0,
        "bad_span": 0,
        "no_probe": 0,
    }
    rows_with_extra_scores = 0

    with source_paths["meta"].open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{source_paths['meta']}:{line_no}: {exc}") from exc
            row = int(rec["row"])
            label = labels.get(row)
            if rec.get("split") != "test":
                skipped["not_test"] += 1
                continue
            if rec.get("status") != "ok":
                skipped["status"] += 1
                continue
            if not rec.get("generated"):
                skipped["not_generated"] += 1
                continue
            if rec.get("truncated"):
                skipped["truncated"] += 1
                continue
            if label is None or not label.get("correct_filtered_b", "").strip():
                skipped["no_label"] += 1
                continue
            logprobs = rec.get("logprobs") or []
            entropies = rec.get("entropies") or []
            if not logprobs or not entropies:
                skipped["no_scores"] += 1
                continue
            answer_tokens = int(rec["pos_c"]) - int(rec["pos_a"])
            if answer_tokens <= 0 or len(logprobs) < answer_tokens or len(entropies) < answer_tokens:
                skipped["bad_span"] += 1
                continue
            if len(logprobs) > answer_tokens or len(entropies) > answer_tokens:
                rows_with_extra_scores += 1

            probe_row = probe.get(row)
            if probe_row is None:
                skipped["no_probe"] += 1
                continue
            y = int(label["correct_filtered_b"])
            if label["split"] != "test" or int(probe_row["y"]) != y:
                raise SystemExit(f"label/split mismatch: {name} row {row}")

            aggregated = aggregate_scores(logprobs, entropies, answer_tokens)
            ys.append(y)
            probe_scores.append(float(probe_row["score"]))
            lengths.append(answer_tokens)
            for key, value in aggregated.items():
                values[key].append(value)

    y = np.asarray(ys, dtype=np.float64)
    probe_score = np.asarray(probe_scores, dtype=np.float64)
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    if not len(y) or y.sum() in (0, len(y)):
        raise SystemExit(f"{name}: test cohort does not contain both classes")

    probe_point = auroc(y, probe_score)
    points = {key: auroc(y, value) for key, value in arrays.items()}

    # Shared resamples make every probe-minus-baseline interval paired by question.
    rng = np.random.default_rng(seed)
    boots = {key: [] for key in METHODS}
    deltas = {key: [] for key in METHODS}
    probe_boots = []
    n = len(y)
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        y_boot = y[index]
        if y_boot.sum() in (0, len(y_boot)):
            continue
        probe_boot = auroc(y_boot, probe_score[index])
        probe_boots.append(probe_boot)
        for key, array in arrays.items():
            baseline_boot = auroc(y_boot, array[index])
            boots[key].append(baseline_boot)
            deltas[key].append(probe_boot - baseline_boot)

    metrics = {}
    for key in METHODS:
        metrics[key] = {
            "label": METHODS[key],
            "auroc": points[key],
            "auroc_ci": interval(boots[key]),
            "probe_minus_baseline": probe_point - points[key],
            "probe_minus_baseline_ci": interval(deltas[key]),
        }

    # These three values must exactly reproduce the immutable primary artifacts.
    baseline_doc = json.loads(source_paths["baseline"].read_text(encoding="utf-8"))
    probe_doc = json.loads(source_paths["probe"].read_text(encoding="utf-8"))
    baseline_test = baseline_doc["splits"]["test"]
    anchors = {
        "first_logprob_diff": points["first_logprob"] - baseline_test["logprob"]["auroc"],
        "first_entropy_diff": points["first_neg_entropy"] - baseline_test["entropy"]["auroc"],
        "probe_diff": probe_point - probe_doc["metrics"]["test"]["auroc"],
    }
    if max(abs(value) for value in anchors.values()) > 1e-12:
        raise SystemExit(f"artifact anchor mismatch for {name}: {anchors}")

    strongest_key = max(METHODS, key=lambda key: points[key])
    return {
        "run": name,
        "model": entry["model"],
        "n": int(n),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "answer_tokens": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "single_token": int(np.sum(np.asarray(lengths) == 1)),
            "max": int(np.max(lengths)),
        },
        "rows_with_extra_generated_scores_after_c": rows_with_extra_scores,
        "skipped": skipped,
        "probe": {
            "auroc": probe_point,
            "auroc_ci": interval(probe_boots),
        },
        "baselines": metrics,
        "strongest_token_baseline_post_hoc": {
            "method": strongest_key,
            "auroc": points[strongest_key],
            "probe_minus_baseline": probe_point - points[strongest_key],
        },
        "anchor_differences": anchors,
        "sources": {key: source_info(path) for key, path in source_paths.items()},
    }


def build_summary(results):
    summary = {}
    count = len(results)
    for key in METHODS:
        deltas = np.asarray([
            run["probe"]["auroc"] - run["baselines"][key]["auroc"]
            for run in results
        ])
        summary[key] = {
            "label": METHODS[key],
            "probe_wins_point_estimate": int(np.sum(deltas > 0)),
            "ties": int(np.sum(deltas == 0)),
            "baseline_wins_point_estimate": int(np.sum(deltas < 0)),
            "delta_min": float(deltas.min()),
            "delta_median": float(np.median(deltas)),
            "delta_max": float(deltas.max()),
            "probe_wins_paired_ci_excludes_zero": int(sum(
                run["baselines"][key]["probe_minus_baseline_ci"][0] > 0
                for run in results
            )),
            "baseline_wins_paired_ci_excludes_zero": int(sum(
                run["baselines"][key]["probe_minus_baseline_ci"][1] < 0
                for run in results
            )),
            "runs": count,
        }
    return summary


def write_csv(path, results):
    fields = (
        ["model", "run", "n", "probe"]
        + list(METHODS)
        + [f"probe_minus_{key}" for key in METHODS]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in results:
            row = {
                "model": run["model"],
                "run": run["run"],
                "n": run["n"],
                "probe": f"{run['probe']['auroc']:.12g}",
            }
            for key in METHODS:
                metric = run["baselines"][key]
                row[key] = f"{metric['auroc']:.12g}"
                row[f"probe_minus_{key}"] = f"{metric['probe_minus_baseline']:.12g}"
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(ROOT / "RUNS.json"))
    parser.add_argument("--out", default=str(ROOT / "pipeline" / "out" / "answer_token_baselines.json"))
    parser.add_argument("--csv-out", default=str(ROOT / "pipeline" / "out" / "answer_token_baselines.csv"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    canonical = [
        (name, entry)
        for name, entry in registry["main_runs"].items()
        if entry.get("canonical") and entry.get("status") == "valid"
    ]
    if not canonical:
        raise SystemExit("registry contains no canonical valid runs")

    results = []
    for name, entry in canonical:
        result = summarize_run(ROOT, name, entry, args.n_boot, args.seed)
        results.append(result)
        baseline = result["baselines"]
        print(
            f"{result['model'][:37]:37s} n={result['n']:4d} "
            f"probe={result['probe']['auroc']:.3f} | "
            f"meanH={baseline['mean_neg_entropy']['auroc']:.3f} "
            f"meanLP={baseline['mean_logprob']['auroc']:.3f} "
            f"minLP={baseline['min_logprob']['auroc']:.3f} "
            f"lastH={baseline['last_neg_entropy']['auroc']:.3f} "
            f"lastLP={baseline['last_logprob']['auroc']:.3f}"
        )

    methods_summary = build_summary(results)
    output = {
        "kind": "exploratory_answer_token_baselines",
        "analysis_date": ANALYSIS_DATE,
        "scope": f"{len(results)} canonical valid TriviaQA runs; held-out test only",
        "label": "correct_filtered_b",
        "cohort": (
            "same exclusion rules as baseline.py/probe.py: status ok, generated, "
            "non-truncated, labeled; joined to saved test probe scores"
        ),
        "answer_span": (
            "first k score entries, k = pos_c - pos_a; required for the "
            "LFM2.5-Thinking fixed 64-score final-generation buffer"
        ),
        "score_direction": "higher means more confidence in correctness; entropy scores are negated",
        "selection_caveat": (
            "These aggregates were inspected after the original test results. "
            "Each listed comparison is exploratory; the post-hoc maximum across "
            "methods is descriptive and has no selection-adjusted confidence interval."
        ),
        "bootstrap": {"n": args.n_boot, "seed": args.seed, "paired_by_question": True},
        "methods": METHODS,
        "methods_summary": methods_summary,
        "runs": results,
        "provenance": {
            "registry": source_info(registry_path),
            "script": source_info(Path(__file__)),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    out_path = Path(args.out)
    csv_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_csv(csv_path, results)

    print("\nSUMMARY")
    for key, summary in methods_summary.items():
        print(
            f"{key:23s}: probe wins {summary['probe_wins_point_estimate']:2d}/{len(results)}, "
            f"paired-significant {summary['probe_wins_paired_ci_excludes_zero']:2d}/{len(results)}, "
            f"delta min/median/max {summary['delta_min']:+.3f}/"
            f"{summary['delta_median']:+.3f}/{summary['delta_max']:+.3f}"
        )
    print(f"\nJSON: {out_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
