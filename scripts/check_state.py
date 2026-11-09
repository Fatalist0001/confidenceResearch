"""Consistency check for the repository state files and the run registry.

This is the "did the paperwork drift from the artifacts" check. It verifies that
RUNS.json still matches the directories in pipeline/out, that the numbers in the
registry equal the numbers inside probe.json/baseline.json/config.json, that the
canonical choice is unambiguous, that deferred work is still marked deferred, and
that both llama.cpp copies still carry identical copies of our five files.

Read-only. Exit code 0 if no hard check failed, 1 otherwise.

Usage:
    python scripts/check_state.py
    python scripts/check_state.py --json C:\\Temp\\state.json
    python scripts/check_state.py --quiet
"""

import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "pipeline", "out")

REQUIRED_FILES = [
    "AGENTS.md",
    "README.md",
    "report.md",
    "PROJECT_STATE.json",
    "RUNS.json",
    "profiles.json",
    "docs/START-HERE.md",
    "docs/SETUP.md",
    "docs/history/AGENTS-full-2026-11-02.md",
    "docs/LFM25-THINKING-PROTOCOL.md",
    "pipeline/requirements.txt",
    "pipeline/gate_report.py",
    "pipeline/preflight.py",
    "scripts/bootstrap.ps1",
    "scripts/build_runs.py",
    "scripts/doctor.py",
    "scripts/run.ps1",
]

TRACKED_FILES = [
    "examples/CMakeLists.txt",
    "examples/hidden-states/CMakeLists.txt",
    "examples/hidden-states/hidden-states.cpp",
    "examples/hidden-states/load_example.py",
    "examples/hidden-states/prompt_dump.cpp",
]

PLAN_FILES = ["HALLUCINATION_PLAN.md", "QUANT_TRANSFER_PLAN.md"]
PLAN_MARKERS = ["NOT AUTHORIZED", "PLAN ONLY", "NOT A TASK"]


class Report:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.fails = 0
        self.warns = 0
        self.checks = []

    def add(self, name, ok, detail="", severity="fail"):
        if not ok:
            if severity == "warn":
                self.warns += 1
            else:
                self.fails += 1
        self.checks.append({"check": name, "ok": bool(ok), "severity": severity, "detail": detail})
        if not (ok and self.quiet):
            mark = "ok" if ok else ("WARN" if severity == "warn" else "FAIL")
            print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
        return ok

    def section(self, title):
        if not self.quiet:
            print(f"\n-- {title}")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def meta_rows(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def read_jsonl_tolerant(path):
    """Return valid object rows and the number of malformed/non-object lines."""
    rows, errors = [], 0
    with open(path, "rb") as f:
        for raw_line in f:
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                errors += 1
                continue
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if not isinstance(value, dict):
                errors += 1
            else:
                rows.append(value)
    return rows, errors


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files(rep):
    rep.section("files")
    for rel in REQUIRED_FILES:
        path = os.path.join(ROOT, rel)
        rep.add(f"exists: {rel}", os.path.exists(path), path if not os.path.exists(path) else "")

    for rel in ("PROJECT_STATE.json", "RUNS.json", "profiles.json"):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        try:
            data = read_json(path)
            rep.add(f"valid JSON: {rel}", isinstance(data, dict), f"{len(data)} top-level keys")
        except json.JSONDecodeError as e:
            rep.add(f"valid JSON: {rel}", False, str(e))


def check_plans(rep):
    rep.section("plan documents")
    for name in PLAN_FILES:
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            rep.add(f"plan exists: {name}", False, path)
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(2000)
        rep.add(
            f"plan is marked not-a-task: {name}",
            any(m in head for m in PLAN_MARKERS),
            "needs a banner in the first 2000 characters",
        )


def check_sources(rep, state):
    rep.section("llama.cpp copies")
    a = os.path.join(ROOT, state["binaries"]["active"]["dir"])
    b = os.path.join(ROOT, state["binaries"]["legacy"]["dir"])
    for label, d in (("active", a), ("legacy", b)):
        rep.add(f"{label} sources present", os.path.isdir(os.path.join(d, "src")),
                "run scripts/bootstrap.ps1" if not os.path.isdir(os.path.join(d, "src")) else d, "warn")
    if not (os.path.isdir(a) and os.path.isdir(b)):
        return
    diffs = []
    for rel in TRACKED_FILES:
        pa, pb = os.path.join(a, rel), os.path.join(b, rel)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            diffs.append(f"{rel} (missing)")
        elif sha256(pa) != sha256(pb):
            diffs.append(rel)
    rep.add("our five files are identical in both copies", not diffs,
            ", ".join(diffs) if diffs else "5 files")

    cmake = os.path.join(a, "examples", "CMakeLists.txt")
    if os.path.exists(cmake):
        text = open(cmake, encoding="utf-8", errors="replace").read()
        rep.add("examples/CMakeLists.txt adds hidden-states",
                text.count("add_subdirectory(hidden-states)") == 1,
                f"{text.count('add_subdirectory(hidden-states)')} occurrences")


def check_registry(rep, runs_doc):
    """RUNS.json rewrites vs the directories on disk."""
    rep.section("registry vs disk")
    main_runs = runs_doc.get("main_runs") or {}
    pilots = runs_doc.get("pilots") or {}
    ad_hoc = runs_doc.get("ad_hoc") or {}
    registered = set(main_runs) | set(pilots) | set(ad_hoc)

    on_disk = sorted(d for d in os.listdir(OUT)
                     if os.path.isdir(os.path.join(OUT, d)) and (
                         d.startswith("main_") or d.startswith("pilot_"))
                     and not d.startswith("pilot20_"))

    missing_from_registry = [d for d in on_disk if d not in registered]
    rep.add("every main_/pilot_ directory is registered", not missing_from_registry,
            ", ".join(missing_from_registry[:6]) if missing_from_registry else f"{len(on_disk)} directories")

    absent_on_disk = [d for d in list(main_runs) + list(pilots)
                      if not os.path.isdir(os.path.join(OUT, d))]
    rep.add("every registered run exists on disk", not absent_on_disk,
            ", ".join(absent_on_disk[:6]) if absent_on_disk
            else f"{len(main_runs) + len(pilots)} main/pilot entries")

    for name, entry in main_runs.items():
        path = os.path.join(OUT, name)
        if not os.path.isdir(path):
            continue
        meta = os.path.join(path, "meta.jsonl")
        cfg_path = os.path.join(path, "config.json")
        if not os.path.exists(meta):
            rep.add(f"{name}: meta.jsonl", False, meta)
            continue
        rows = meta_rows(meta)
        rep.add(f"{name}: meta_rows", rows == entry.get("meta_rows"),
                f"{rows} on disk vs {entry.get('meta_rows')} in registry")
        cfg = read_json(cfg_path) if os.path.exists(cfg_path) else None

        shape = entry.get("shape")
        if shape:
            want0 = entry.get("meta_rows")
            rep.add(f"{name}: shape rows", shape[0] == want0,
                    f"shape[0]={shape[0]} meta_rows={want0}")
            if cfg is not None:
                rep.add(f"{name}: shape layers/embd",
                        shape[1] == cfg.get("n_layers_selected") and shape[2] == cfg.get("n_embd"),
                        f"{shape[1:]} vs [{cfg.get('n_layers_selected')}, {cfg.get('n_embd')}]")
        if cfg is not None:
            binary = entry.get("binary") or {}
            rep.add(f"{name}: binary commit/build",
                    binary.get("commit") == cfg.get("llama_cpp_commit")
                    and binary.get("build") == cfg.get("llama_cpp_build"),
                    f"registry {binary} vs config "
                    f"{cfg.get('llama_cpp_commit')}/{cfg.get('llama_cpp_build')}")

        labels = entry.get("labels") or {}
        evaluation = entry.get("evaluation") or {}
        label_column = evaluation.get("label_column", "correct_filtered_b")
        required_splits = evaluation.get("required_splits", ["train", "val", "test"])
        labels_path = os.path.join(path, "labels.csv")
        if labels.get("rows") is not None:
            rep.add(f"{name}: labels.csv exists", os.path.exists(labels_path), labels_path)
            if os.path.exists(labels_path):
                label_rows = read_csv_rows(labels_path)
                values = [str(r.get(label_column, "")).strip() for r in label_rows]
                labeled = sum(v in ("0", "1") for v in values)
                nulls = len(label_rows) - labeled
                pos = sum(v == "1" for v in values)
                base_rate = round(pos / labeled, 4) if labeled else None
                disk_summary = (len(label_rows), labeled, nulls, base_rate)
                registry_summary = (labels.get("rows"), labels.get("labeled"),
                                    labels.get("null"), labels.get("base_rate_filtered_b") if label_column == "correct_filtered_b" else labels.get("base_rate"))
                rep.add(f"{name}: labels summary", disk_summary == registry_summary,
                        f"disk {disk_summary} vs registry {registry_summary}")
                if entry.get("status") == "valid":
                    bad_splits = []
                    for split in required_splits:
                        split_values = [str(r.get(label_column, "")).strip()
                                         for r in label_rows if r.get("split") == split
                                         and str(r.get(label_column, "")).strip() in ("0", "1")]
                        if set(split_values) != {"0", "1"}:
                            bad_splits.append(split)
                    rep.add(f"{name}: both label classes in {'/'.join(required_splits)}", not bad_splits,
                            str(bad_splits))
                meta_objects, _ = read_jsonl_tolerant(meta)
                truncated = sum(bool(r.get("truncated")) for r in meta_objects)
                rep.add(f"{name}: labels truncated count", truncated == labels.get("truncated"),
                        f"{truncated} on disk vs {labels.get('truncated')} in registry")

        probe = entry.get("probe")
        probe_path = os.path.join(path, "probe.json")
        if probe:
            rep.add(f"{name}: probe.json exists", os.path.exists(probe_path), probe_path)
            if os.path.exists(probe_path):
                p = read_json(probe_path)
                sel = p.get("selected") or {}
                metrics = (p.get("metrics") or {}).get("test") or {}
                pair = (round(sel.get("val_auroc", -1), 4), round(metrics.get("auroc", -1), 4))
                rep.add(f"{name}: probe val/test match probe.json",
                        pair == (probe.get("val_auroc"), probe.get("test_auroc")),
                        f"disk {pair} vs registry ({probe.get('val_auroc')}, {probe.get('test_auroc')})")
                rep.add(f"{name}: probe lambda/layer match",
                        p.get("selected_lambda") == probe.get("lambda")
                        and sel.get("point") == probe.get("point")
                        and sel.get("layer") == probe.get("layer")
                        and sel.get("layer_id", sel.get("layer")) == probe.get("layer_id"),
                        f"disk {p.get('selected_lambda')}/{sel.get('point')}/{sel.get('layer')}"
                        f"/{sel.get('layer_id', sel.get('layer'))} vs registry "
                        f"{probe.get('lambda')}/{probe.get('point')}/{probe.get('layer')}/{probe.get('layer_id')}")
                model_npz = os.path.join(path, "probe_model.npz")
                rep.add(f"{name}: probe_model.npz", os.path.exists(model_npz),
                        "run save_probe_model.py --dir <run>" if not os.path.exists(model_npz) else "",
                        "warn")
        elif entry.get("probe_status") in ("complete",):
            rep.add(f"{name}: probe_status=complete but no probe numbers", False,
                    "registry claims a probe that is not stored")

        bl_path = os.path.join(path, "baseline.json")
        bl = entry.get("baseline_test_auroc")
        if bl:
            if os.path.exists(bl_path):
                b = read_json(bl_path)
                test = (b.get("splits") or {}).get("test") or {}
                pairs = {
                    "logprob": round(((test.get("logprob") or {}).get("auroc") or -1), 4),
                    "entropy": round(((test.get("entropy") or {}).get("auroc") or -1), 4),
                }
                rep.add(f"{name}: baseline test AUROC matches",
                        pairs["logprob"] == bl.get("logprob") and pairs["entropy"] == bl.get("entropy"),
                        f"disk {pairs} vs registry {bl}")
            else:
                rep.add(f"{name}: baseline.json exists", False, bl_path)

        gate = os.path.join(path, "gate.json")
        if entry.get("status") == "partial":
            rep.add(f"{name}: partial run has no probe", probe is None, "probe numbers must not exist")
        verify_path = os.path.join(path, "verify.json")
        verify = entry.get("verify") or {}
        if os.path.exists(verify_path):
            disk_verify = read_json(verify_path)
            expected_status = "passed" if int(disk_verify.get("fails", 0)) == 0 else "failed"
            rep.add(f"{name}: verify summary",
                    verify.get("status") == expected_status
                    and verify.get("fails") == disk_verify.get("fails")
                    and verify.get("warns") == disk_verify.get("warns"),
                    f"disk {expected_status}/{disk_verify.get('fails')}/{disk_verify.get('warns')} "
                    f"vs registry {verify}")
        else:
            rep.add(f"{name}: verify status not_run", verify.get("status") == "not_run",
                    str(verify))
        if os.path.exists(gate):
            g = read_json(gate)
            if entry.get("status") == "valid" and not g.get("passed", False):
                rep.add(f"{name}: valid status but gate.json failed", False,
                        f"{g.get('fails')} failed checks")
        if entry.get("status") == "invalid":
            mark = os.path.join(path, "README-invalid.md")
            rep.add(f"{name}: invalid run is labelled", os.path.exists(mark),
                     "add README-invalid.md", "warn")

        if evaluation.get("kind") == "hotpotqa_frozen_transfer":
            transfer_path = os.path.join(path, "transfer.json")
            rep.add(f"{name}: frozen transfer exists", os.path.exists(transfer_path), transfer_path)
            rep.add(f"{name}: no target fitting artifacts", not any(os.path.exists(os.path.join(path, f))
                    for f in ("probe.json", "probe_model.npz", "probe2.json", "baseline.json")))
            if os.path.exists(transfer_path):
                transfer = read_json(transfer_path)
                summary = entry.get("transfer") or {}
                rep.add(f"{name}: transfer summary matches artifacts",
                        bool(summary) and all(summary[key] == transfer.get(key) for key in summary))
                feature = transfer.get("frozen_feature") or {}
                rep.add(f"{name}: frozen feature is not fitted on target",
                        feature.get("point") == "C" and feature.get("layer_id") == 13 and
                        feature.get("target_array_index") == 0 and feature.get("lambda") == 1000.0 and
                        feature.get("fit_on_target") is False and feature.get("calibration") is False)
                source = transfer.get("source_probe") or {}
                source_path = source.get("path", "")
                if not os.path.isfile(source_path):
                    # Fingerprints retain the original machine path; git clones do not.
                    recorded = source_path.replace("\\", "/")
                    source_path = os.path.join(OUT, os.path.basename(os.path.dirname(recorded)), os.path.basename(recorded))
                rep.add(f"{name}: source probe hash is unchanged", os.path.isfile(source_path) and
                        sha256(source_path) == source.get("sha256"))
                score_path = os.path.join(path, "transfer_scores.csv")
                if rep.add(f"{name}: transfer scores exist", os.path.exists(score_path)):
                    score_rows = read_csv_rows(score_path)
                    primary_n = (transfer.get("primary") or {}).get("n")
                    rep.add(f"{name}: transfer scores cover all attempts",
                            len(score_rows) == rows and all(r.get("split") == "test" for r in score_rows))
                    rep.add(f"{name}: same primary rows for all methods",
                            sum(r.get("analysis_included") == "1" for r in score_rows) == primary_n and
                            all(all(bool(r.get(key, "")) == (r.get("analysis_included") == "1")
                                    for key in ("frozen_probe_logit", "first_token_logprob", "negative_first_token_entropy"))
                                for r in score_rows))

    for name, entry in pilots.items():
        path = os.path.join(OUT, name)
        if not os.path.isdir(path):
            continue
        meta_path = os.path.join(path, "meta.jsonl")
        cfg_path = os.path.join(path, "config.json")
        if not (os.path.exists(meta_path) and os.path.exists(cfg_path)):
            rep.add(f"{name}: config/meta exist", False, path)
            continue
        rows = meta_rows(meta_path)
        cfg = read_json(cfg_path)
        meta_objects, malformed = read_jsonl_tolerant(meta_path)
        truncated = sum(bool(r.get("truncated")) for r in meta_objects)
        rep.add(f"{name}: meta_rows", rows == entry.get("meta_rows"),
                f"{rows} vs {entry.get('meta_rows')}")
        rep.add(f"{name}: meta JSON is readable", malformed == 0,
                f"{malformed} malformed rows")
        rep.add(f"{name}: truncated", truncated == entry.get("truncated"),
                f"{truncated} vs {entry.get('truncated')}")
        expected_shape = [rows, cfg.get("n_layers_selected"), cfg.get("n_embd")]
        rep.add(f"{name}: shape", entry.get("shape") == expected_shape,
                f"{entry.get('shape')} vs {expected_shape}")
        rep.add(f"{name}: never citable", entry.get("citable") is False,
                str(entry.get("citable")))
        if entry.get("probe_status") == "pipeline_check":
            rep.add(f"{name}: pipeline-check probe exists",
                    os.path.exists(os.path.join(path, "probe.json")), path)


def check_canonical(rep, runs_doc):
    rep.section("canonical runs")
    main_runs = runs_doc.get("main_runs") or {}
    # A family groups sizes (qwen3.5 = 0.8B/2B/4B/9B), so the citation unit is the
    # model string, not the family.
    by_model = {}
    for name, entry in main_runs.items():
        if entry.get("canonical"):
            by_model.setdefault(entry.get("model"), []).append(name)
    for model, names in sorted(by_model.items(), key=lambda kv: str(kv[0])):
        rep.add(f"one canonical run for {model}", len(names) == 1, ", ".join(names))

    for name, entry in main_runs.items():
        if not entry.get("canonical"):
            continue
        rep.add(f"{name}: canonical is valid", entry.get("status") == "valid", entry.get("status", ""))
        rep.add(f"{name}: canonical is not superseded", not entry.get("superseded_by"),
                str(entry.get("superseded_by")))
        rep.add(f"{name}: canonical is usable", bool(entry.get("usable_for_analysis")), "")
    for name, entry in main_runs.items():
        if entry.get("status") in ("invalid", "partial"):
            rep.add(f"{name}: not citable", entry.get("probe_status") == "not_citable",
                    str(entry.get("probe_status")), "warn")
            rep.add(f"{name}: has no probe numbers", entry.get("probe") is None,
                    "invalid runs must not carry probe metrics")


def check_profiles(rep, profiles_doc, state):
    rep.section("profiles")
    profiles = profiles_doc.get("profiles") or {}
    rep.add("profiles exist", len(profiles) >= 10, f"{len(profiles)} profiles")

    valid_binaries = {"active", "legacy", "kaggle"}
    valid_gate_keys = set((profiles_doc.get("defaults") or {}).get("pilot_gate") or {})
    valid_flag_keys = {
        "generate", "max_new_tokens", "max_len", "layers", "ngl",
        "no_template", "no_thinking", "thinking", "batch_rows",
        "temperature", "top_k", "repeat_penalty", "seed",
    }
    default_flags = (profiles_doc.get("defaults") or {}).get("flags") or {}
    rep.add("default flag keys are known", not (set(default_flags) - valid_flag_keys),
            ", ".join(sorted(set(default_flags) - valid_flag_keys)))
    for name, p in profiles.items():
        rep.add(f"{name}: binary key", p.get("binary") in valid_binaries, str(p.get("binary")))
        rep.add(f"{name}: has inputs pilot+main",
                set((p.get("inputs") or {})).issuperset({"pilot", "main"}),
                ", ".join(sorted((p.get("inputs") or {}))))
        unknown = set(p.get("pilot_gate") or {}) - valid_gate_keys
        rep.add(f"{name}: gate keys are known", not unknown, ", ".join(sorted(unknown)))
        unknown_flags = set(p.get("flags") or {}) - valid_flag_keys
        rep.add(f"{name}: flag keys are known", not unknown_flags,
                ", ".join(sorted(unknown_flags)))
        merged_flags = dict(default_flags)
        merged_flags.update(p.get("flags") or {})
        rep.add(f"{name}: thinking flags do not conflict",
                not (merged_flags.get("thinking") and merged_flags.get("no_thinking")), "")
        blocked = p.get("blocked")
        if blocked:
            ids = {b.get("id") for b in state["authorization"]["blocked_actions"]}
            rep.add(f"{name}: blocked id is in PROJECT_STATE", blocked.get("id") in ids, str(blocked.get("id")))

    # Every family used by the registry should have a profile.
    used = {e.get("family") for e in (read_json(os.path.join(ROOT, "RUNS.json"))
                                      .get("main_runs") or {}).values()}
    applies = {a for p in profiles.values() for a in (p.get("applies_to") or [])}
    rep.add("every registry family has a profile", used <= applies,
            ", ".join(sorted(x for x in used - applies if x)), "warn")


def check_state_file(rep, state):
    rep.section("project state")
    bins = state.get("binaries") or {}
    for label in ("active", "legacy"):
        entry = bins.get(label) or {}
        rep.add(f"{label} binary path is recorded", bool(entry.get("exe")), entry.get("exe", ""))
        rep.add(f"{label} commit/build is recorded", bool(entry.get("commit")) and bool(entry.get("build")),
                f"{entry.get('commit')} / {entry.get('build')}")
    ids = [b.get("id") for b in state["authorization"]["blocked_actions"]]
    rep.add("blocked actions have unique ids", len(ids) == len(set(ids)), f"{len(ids)} entries")
    protocol = state.get("protocol") or {}
    rep.add("protocol lambdas are the frozen grid", protocol.get("lambdas") == [100, 300, 1000, 3000],
            str(protocol.get("lambdas")))
    rep.add("protocol uses correct_filtered_b", protocol.get("label_column") == "correct_filtered_b",
            str(protocol.get("label_column")))


def check_data(rep):
    rep.section("inputs")
    for name, want in (("main.jsonl", 6000), ("pilot.jsonl", 200)):
        path = os.path.join(ROOT, "pipeline", "data", name)
        if not os.path.exists(path):
            rep.add(f"data/{name} exists", False, path)
            continue
        rows = meta_rows(path)
        rep.add(f"data/{name} rows", rows == want, f"{rows} vs {want}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true", help="print failures and warnings only")
    args = ap.parse_args()

    rep = Report(quiet=args.quiet)
    state_path = os.path.join(ROOT, "PROJECT_STATE.json")
    runs_path = os.path.join(ROOT, "RUNS.json")
    profiles_path = os.path.join(ROOT, "profiles.json")
    for p in (state_path, runs_path, profiles_path):
        if not os.path.exists(p):
            print(f"[FAIL] missing {p}")
            return 1

    state = read_json(state_path)
    runs_doc = read_json(runs_path)
    profiles_doc = read_json(profiles_path)

    check_files(rep)
    check_plans(rep)
    check_sources(rep, state)
    check_registry(rep, runs_doc)
    check_canonical(rep, runs_doc)
    check_profiles(rep, profiles_doc, state)
    check_state_file(rep, state)
    check_data(rep)

    report = {
        "root": ROOT,
        "passed": rep.fails == 0,
        "fails": rep.fails,
        "warns": rep.warns,
        "checks": rep.checks,
    }
    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"\ncheck_state: {rep.fails} failed, {rep.warns} warnings")
    if rep.fails:
        print("A failed check means the paper trail and the artifacts disagree. "
              "Fix the file, never the artifact.")
    return 0 if rep.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
