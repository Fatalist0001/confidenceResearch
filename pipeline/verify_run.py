"""Check that a run directory is internally consistent and readable.

Verifies the contract between config.json, meta.jsonl, labels.csv and the two
hidden-state arrays, so that a broken run (all-zero rows, shifted indices,
missing labels) fails here instead of turning into a silent AUROC of 0.5.

Usage:
    python verify_run.py --dir out/main_nemotron3nano4b
    python verify_run.py --dir out/main_qwen08 --json out/main_qwen08/verify.json
    python verify_run.py --dir out/main_x --full          # scan every row

Exit code 0 if all checks pass, 1 otherwise.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


class Report:
    def __init__(self):
        self.fails = 0
        self.warns = 0
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.fails += 1
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.warns += 1
        self.checks.append({"check": name, "ok": None, "detail": detail})
        print(f"[warn] {name}" + (f": {detail}" if detail else ""))


def read_jsonl(path):
    """Records plus a list of malformed lines (interrupted runs leave those)."""
    records, errors = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append({"line": i, "error": str(e)})
                continue
            if not isinstance(rec, dict):
                errors.append({"line": i, "error": "not a JSON object"})
                continue
            records.append(rec)
    return records, errors


def read_labels(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["row"]): r for r in csv.DictReader(f)}


def input_dup_ids(cfg, run_dir):
    """Duplicate ids of the batch file named in config.json, or None if not found.

    main.jsonl contains two different questions with the same id (bb_956), so a
    duplicated id is only an error if the input does not have it either.
    """
    path = cfg.get("batch")
    if not path:
        return None
    base = os.path.abspath(run_dir)
    cands = [path]
    for _ in range(4):
        cands.append(os.path.join(base, path))
        base = os.path.dirname(base)
    for cand in cands:
        if not os.path.isfile(cand):
            continue
        counts = {}
        with open(cand, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(rec.get("id", ""))
                counts[key] = counts.get(key, 0) + 1
        return {k for k, v in counts.items() if v > 1}
    return None


def sample_rows(n, k, seed=0):
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    picked = set(rng.choice(n, size=k, replace=False).tolist())
    picked.update(range(min(4, n)))
    picked.update(range(max(0, n - 4), n))
    return sorted(picked)


def row_stats(arr, rows):
    """max|value| and finite flag per sampled row."""
    out = []
    for i in rows:
        block = np.asarray(arr[i], dtype=np.float32)
        out.append((float(np.abs(block).max()) if block.size else 0.0, bool(np.isfinite(block).all())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--sample", type=int, default=32, help="rows to inspect in detail")
    ap.add_argument("--full", action="store_true", help="check every row instead of a sample")
    ap.add_argument("--allow-partial", action="store_true", help="accept meta with fewer rows than n_examples")
    args = ap.parse_args()

    d = args.dir
    rep = Report()

    cfg_path = os.path.join(d, "config.json")
    meta_path = os.path.join(d, "meta.jsonl")
    a_path = os.path.join(d, "hidden_a.npy")
    c_path = os.path.join(d, "hidden_c.npy")
    lab_path = os.path.join(d, "labels.csv")

    for p in (cfg_path, meta_path, a_path, c_path):
        rep.check(f"file {os.path.basename(p)}", os.path.exists(p), p)
    if rep.fails:
        return 1

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    layers = [int(x) for x in cfg.get("layers", [])]
    n_examples = int(cfg.get("n_examples", 0))
    n_embd = int(cfg.get("n_embd", 0))

    rep.check("config: layers == n_layers_selected", len(layers) == int(cfg.get("n_layers_selected", -1)),
              f"{len(layers)} vs {cfg.get('n_layers_selected')}")
    rep.check("config: layers unique and in range",
              len(set(layers)) == len(layers) and all(0 <= l < int(cfg.get("n_layers", 0)) for l in layers),
              str(layers))
    if "layers_requested" in cfg:
        rep.warn("config: layers were pruned by the l_out probe",
                 f"{len(cfg['layers_requested'])} requested -> {len(layers)} with l_out")

    meta, meta_errors = read_jsonl(meta_path)
    if meta_errors:
        detail = f"{len(meta_errors)} malformed line(s): {meta_errors[:3]}"
        if args.allow_partial:
            rep.warn("meta: malformed JSON lines", detail)
        else:
            rep.check("meta: every line is valid JSON", False, detail)
    else:
        rep.check("meta: every line is valid JSON", True, f"{len(meta)} lines")
    rep.check("meta: not empty", len(meta) > 0, f"{len(meta)} lines")
    if len(meta) != n_examples:
        detail = f"{len(meta)} of {n_examples} expected"
        if args.allow_partial:
            rep.warn("meta: incomplete run", detail)
        else:
            rep.check("meta: all examples processed", False, detail)

    rows = [int(r["row"]) for r in meta]
    missing = sorted(set(range(max(rows) + 1)) - set(rows)) if rows else []
    if missing or len(set(rows)) != len(rows):
        detail = f"first missing/duplicate rows: {missing[:10]}"
        if args.allow_partial:
            rep.warn("meta: row indices not contiguous", detail)
        else:
            rep.check("meta: row indices contiguous from 0", False, detail)
    else:
        rep.check("meta: row indices contiguous from 0", True, f"{len(rows)} rows")
    ids = [r.get("id", "") for r in meta]
    dup_counts = {}
    for i in ids:
        dup_counts[i] = dup_counts.get(i, 0) + 1
    dups = sorted(k for k, v in dup_counts.items() if v > 1)
    if not dups:
        rep.check("meta: ids unique", True, f"{len(ids)} ids")
    else:
        known = input_dup_ids(cfg, d)
        if known is not None and set(dups) <= known:
            rep.check("meta: duplicate ids match the input file", True,
                      f"{len(dups)} duplicated id(s) also duplicated in {cfg.get('batch')}: {dups}")
        else:
            rep.check("meta: ids unique", False,
                      f"{len(dups)} duplicated id(s): {dups}"
                      + ("" if known is not None else " (input file not found, cannot confirm)"))

    splits = {}
    for r in meta:
        splits[r.get("split", "?")] = splits.get(r.get("split", "?"), 0) + 1
    rep.check("meta: all rows have a split", all(k in ("train", "val", "test") for k in splits), str(splits))

    status = {}
    for r in meta:
        status[r.get("status", "?")] = status.get(r.get("status", "?"), 0) + 1
    n_ok = status.get("ok", 0)
    rep.check("meta: at least one ok row", n_ok > 0, str(status))
    if status.get("skipped", 0):
        reasons = {}
        for r in meta:
            if r.get("status") != "ok":
                reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
        rep.warn("meta: skipped rows", f"{status.get('skipped', 0)}: {reasons}")

    n_gen = sum(1 for r in meta if r.get("generated"))
    n_trunc = sum(1 for r in meta if r.get("truncated"))
    print(f"       splits={splits} status={status} generated={n_gen} truncated={n_trunc}")
    if n_gen == 0:
        rep.warn("meta: no generated answers (labels will be empty unless the answer was in the input)")

    bad_meta = []
    for r in meta:
        if r.get("status") != "ok":
            continue
        if r.get("truncated"):
            continue
        pos_a, pos_c, n_tok = r.get("pos_a"), r.get("pos_c"), r.get("n_tokens")
        if not isinstance(pos_a, int) or not isinstance(pos_c, int):
            bad_meta.append((r.get("row"), "pos is not an int"))
        elif not (0 <= pos_a <= pos_c):
            bad_meta.append((r.get("row"), f"pos_a={pos_a} pos_c={pos_c}"))
        elif isinstance(n_tok, int) and n_tok != pos_c + 1:
            bad_meta.append((r.get("row"), f"n_tokens={n_tok} pos_c={pos_c}"))
    rep.check("meta: ok rows have consistent pos_a/pos_c/n_tokens", not bad_meta, str(bad_meta[:5]))

    arr_a = np.load(a_path, mmap_mode="r")
    arr_c = np.load(c_path, mmap_mode="r")
    want = (n_examples, len(layers), n_embd)
    shape_ok = True
    for name, arr in (("hidden_a.npy", arr_a), ("hidden_c.npy", arr_c)):
        ok = arr.shape == want
        shape_ok = shape_ok and ok
        rep.check(f"{name}: shape {want}", ok, str(arr.shape))
        ok = arr.dtype == np.float16
        shape_ok = shape_ok and ok
        rep.check(f"{name}: dtype float16", ok, str(arr.dtype))
    ok = arr_a.shape == arr_c.shape
    shape_ok = shape_ok and ok
    rep.check("hidden_a.npy and hidden_c.npy have the same shape", ok, f"{arr_a.shape} vs {arr_c.shape}")
    if not shape_ok:
        return 1

    n_scan = len(rows)
    scan = rows if args.full else [rows[i] for i in sample_rows(n_scan, args.sample)]
    meta_by_row = {int(r["row"]): r for r in meta}
    sa = dict(zip(scan, row_stats(arr_a, scan)))
    sc = dict(zip(scan, row_stats(arr_c, scan)))

    ok_rows = [i for i in scan if meta_by_row[i].get("status") == "ok" and not meta_by_row[i].get("truncated")]
    zero_ok = [i for i in ok_rows if sa[i][0] == 0.0 or sc[i][0] == 0.0]
    rep.check("hidden states are non-zero on ok rows (sampled)", not zero_ok,
              f"{len(zero_ok)} of {len(ok_rows)} sampled rows are all-zero; sample {zero_ok[:5]}")

    skipped_rows = [i for i in scan if meta_by_row[i].get("status") != "ok"]
    nonzero_skipped = [i for i in skipped_rows if sa[i][0] != 0.0 or sc[i][0] != 0.0]
    rep.check("hidden states are zero on skipped rows (sampled)", not nonzero_skipped,
              f"{nonzero_skipped[:5]}")

    nonfinite = [i for i in scan if not sa[i][1] or not sc[i][1]]
    rep.check("no NaN/Inf in sampled rows", not nonfinite, str(nonfinite[:5]))

    zero_layers = []
    for l in range(arr_a.shape[1]):
        block = np.asarray(arr_a[np.asarray(scan), l, :], dtype=np.float32)
        if block.size and float(np.abs(block).max()) == 0.0:
            zero_layers.append(layers[l])
    rep.check("every layer has non-zero values (sampled)", not zero_layers, f"zero layers: {zero_layers}")

    same_pos = [i for i in ok_rows if meta_by_row[i].get("pos_a") == meta_by_row[i].get("pos_c")
                and not meta_by_row[i].get("generated")][:8]
    if same_pos:
        bad = [i for i in same_pos if not np.array_equal(np.asarray(arr_a[i]), np.asarray(arr_c[i]))]
        rep.check("A == C when there is no answer (sampled)", not bad, str(bad))

    if os.path.exists(lab_path):
        labels = read_labels(lab_path)
        rep.check("labels: one row per meta row", len(labels) == len(meta),
                  f"{len(labels)} vs {len(meta)}")
        mismatch = [r for r in meta
                    if int(r["row"]) in labels and labels[int(r["row"])]["split"].strip()
                    and labels[int(r["row"])]["split"] != r.get("split")]
        rep.check("labels: split matches meta", not mismatch,
                  str([m["row"] for m in mismatch[:5]]))
        labeled = [r for r in meta if int(r["row"]) in labels
                   and labels[int(r["row"])]["correct_filtered_b"].strip() != ""]
        pos = sum(1 for r in labeled if labels[int(r["row"])]["correct_filtered_b"] == "1")
        print(f"       labeled={len(labeled)} positives={pos}"
              f" ({pos / len(labeled):.3f})" if labeled else "       labeled=0")
        if labeled:
            rep.check("labels: both classes present", 0 < pos < len(labeled), f"{pos}/{len(labeled)}")
            by_split = {}
            for r in labeled:
                s = r.get("split", "?")
                by_split.setdefault(s, [0, 0])
                by_split[s][0] += 1
                by_split[s][1] += labels[int(r["row"])]["correct_filtered_b"] == "1"
            print("       per split (rows/positives): "
                  + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_split.items())))
            empty_splits = [s for s in ("train", "val", "test") if by_split.get(s, [0])[0] == 0]
            rep.check("labels: train/val/test all non-empty", not empty_splits, str(empty_splits))
        else:
            rep.check("labels: at least one usable label", False, "all correct_filtered_b are empty")
    else:
        rep.warn("labels.csv missing", lab_path)

    result = {"dir": d.replace("\\", "/"), "fails": rep.fails, "warns": rep.warns, "checks": rep.checks}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"written: {args.json}")

    print()
    print(f"{'FAILED' if rep.fails else 'OK'}: {rep.fails} failed, {rep.warns} warning(s)")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
