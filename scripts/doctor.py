"""Environment check for llama-hidden-states.

Answers one question: can this machine run the workflow right now, and what is
missing. Read-only: writes nothing except an optional --json report.

Usage:
    python scripts/doctor.py
    python scripts/doctor.py --model D:\\models\\x.gguf --profile qwen
    python scripts/doctor.py --input pipeline/data/main.jsonl --json C:\\Temp\\doctor.json

Exit code 0 if no hard check failed, 1 otherwise.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(ROOT, "pipeline")
OUT = os.path.join(PIPELINE, "out")

TRACKED_FILES = [
    "examples/CMakeLists.txt",
    "examples/hidden-states/CMakeLists.txt",
    "examples/hidden-states/hidden-states.cpp",
    "examples/hidden-states/load_example.py",
    "examples/hidden-states/prompt_dump.cpp",
]


class Report:
    def __init__(self):
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
        mark = "ok" if ok else ("WARN" if severity == "warn" else "FAIL")
        print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
        return ok


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def run_version(exe, timeout=60):
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=timeout, errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return None, str(e)


def check_binary(rep, entry, label, required):
    exe = os.path.join(ROOT, entry.get("exe", ""))
    if not os.path.exists(exe):
        rep.add(f"{label} binary exists", False, f"{exe} (run scripts/bootstrap.ps1 -Build)", "fail" if required else "warn")
        return
    rep.add(f"{label} binary exists", True, exe)
    code, out = run_version(exe)
    if code != 0:
        rep.add(f"{label} binary runs", False, f"exit {code}: {out.strip()[:200]}")
        return
    # Newer builds print "build 12033, commit 785322d17"; the old pin prints
    # "version: 9869 (d4cff114c)" because it has no build-info header.
    m = re.search(r"build (\d+), commit ([0-9a-f]+)", out)
    if m:
        build, commit = int(m.group(1)), m.group(2)
    else:
        m = re.search(r"version:\s*(\d+)\s+\(([0-9a-f]+)\)", out)
        if not m:
            rep.add(f"{label} version is readable", False, out.strip()[:200], "warn")
            return
        build, commit = int(m.group(1)), m.group(2)
    want_build, want_commit = int(entry.get("build", -1)), str(entry.get("commit", ""))
    rep.add(
        f"{label} version matches PROJECT_STATE.json",
        build == want_build and commit == want_commit,
        f"got build {build} commit {commit}, want build {want_build} commit {want_commit}",
    )
    dump = os.path.join(ROOT, entry.get("prompt_dump", ""))
    rep.add(f"{label} hs-prompt-dump exists", os.path.exists(dump), dump, "fail" if required else "warn")


def check_model(rep, model, profile):
    if not model:
        return
    path = model if os.path.isabs(model) else os.path.join(ROOT, model)
    if not os.path.exists(path):
        rep.add("model file exists", False, path)
        return
    size_gb = os.path.getsize(path) / (1024 ** 3)
    rep.add("model file exists", True, f"{size_gb:.2f} GB")

    state = read_json(os.path.join(ROOT, "PROJECT_STATE.json"))
    dump = os.path.join(ROOT, state["binaries"]["active"]["prompt_dump"])
    if not os.path.exists(dump):
        rep.add("prompt dump for the model", False, "hs-prompt-dump missing", "warn")
        return
    try:
        p = subprocess.run([dump, path], capture_output=True, text=True, timeout=300, errors="replace")
    except Exception as e:
        rep.add("prompt dump for the model", False, str(e))
        return
    out = (p.stdout or "") + (p.stderr or "")
    rep.add("prompt dump for the model", p.returncode == 0, f"exit {p.returncode}")
    if p.returncode != 0:
        return
    m = re.search(r"add_bos=(\d+)\s+bos=(-?\d+)", out)
    first = re.search(r"tokens add_special=true\s+\((\d+)\):\s+(-?\d+)", out)
    if not m:
        rep.add("prompt dump reports add_bos", False, "unparsable output", "warn")
        return
    add_bos, bos = int(m.group(1)), int(m.group(2))
    first_tok = int(first.group(2)) if first else None
    rep.add("prompt dump reports add_bos", True, f"add_bos={add_bos} bos={bos} first_token={first_tok}")
    if add_bos:
        rep.add("required BOS is the first prompt token", first_tok == bos,
                f"BOS id {bos}, first token {first_tok}")

    if profile:
        want = profile.get("prompt_expectation") or {}
        want_bos = want.get("add_bos")
        want_first = want.get("first_token")
        if want_bos is not None:
            rep.add("model matches the profile prompt regime", int(want_bos) == add_bos,
                    f"profile expects add_bos={want_bos}, model reports {add_bos}")
        if want_first is not None and first_tok is not None:
            rep.add("first prompt token is the expected BOS", int(want_first) == first_tok,
                    f"profile expects {want_first}, got {first_tok}")


def check_input(rep, path):
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(path):
        rep.add("input file exists", False, path)
        return
    rows = count_lines(path)
    rep.add("input file exists", True, f"{rows} rows, {os.path.getsize(path) / 1024:.0f} KB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rep = Report()
    state = read_json(os.path.join(ROOT, "PROJECT_STATE.json"))
    profiles = read_json(os.path.join(ROOT, "profiles.json"))
    profile = None
    if args.profile:
        props = profiles.get("profiles", {})
        if args.profile not in props:
            print(f"[FAIL] unknown profile '{args.profile}'. Valid: {', '.join(sorted(props))}")
            return 1
        profile = props[args.profile]

    # ----- python -----
    v = sys.version_info
    rep.add("python >= 3.11", v >= (3, 11), f"{v.major}.{v.minor}.{v.micro} at {sys.executable}")
    for module in ("numpy", "datasets"):
        try:
            mod = __import__(module)
            rep.add(f"python package {module}", True, getattr(mod, "__version__", "unknown"),
                    "fail" if module == "numpy" else "warn")
        except ImportError:
            rep.add(f"python package {module}", False,
                    "pip install -r pipeline/requirements.txt",
                    "fail" if module == "numpy" else "warn")
    venv = os.path.join(PIPELINE, ".venv", "Scripts", "python.exe")
    rep.add("pipeline virtualenv exists", os.path.exists(venv), venv, "warn")

    # ----- binaries -----
    check_binary(rep, state["binaries"]["active"], "active", required=True)
    check_binary(rep, state["binaries"]["legacy"], "legacy", required=False)

    # ----- llama.cpp sources -----
    for entry in (state["binaries"]["active"], state["binaries"]["legacy"]):
        d = os.path.join(ROOT, entry["dir"])
        src = os.path.join(d, "src", "llama-context.cpp")
        rep.add(f"llama.cpp sources: {entry['dir']}", os.path.exists(src),
                "run scripts/bootstrap.ps1" if not os.path.exists(src) else "", "warn")
    a = os.path.join(ROOT, state["binaries"]["active"]["dir"])
    b = os.path.join(ROOT, state["binaries"]["legacy"]["dir"])
    if os.path.isdir(a) and os.path.isdir(b):
        import hashlib

        def digest(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        diffs = []
        for rel in TRACKED_FILES:
            pa, pb = os.path.join(a, rel), os.path.join(b, rel)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                diffs.append(f"{rel} (missing)")
            elif digest(pa) != digest(pb):
                diffs.append(rel)
        rep.add("our files are identical in both llama.cpp copies", not diffs, ", ".join(diffs) if diffs else "5 files")

    # ----- data -----
    for name, want in (("main.jsonl", 6000), ("pilot.jsonl", 200)):
        path = os.path.join(PIPELINE, "data", name)
        if not os.path.exists(path):
            rep.add(f"data/{name}", False, "missing")
            continue
        rows = count_lines(path)
        rep.add(f"data/{name}", rows == want, f"{rows} rows (expected {want})")
    if args.input:
        check_input(rep, args.input)

    # ----- output directory and disk -----
    rep.add("pipeline/out exists", os.path.isdir(OUT), OUT, "warn")
    try:
        os.makedirs(OUT, exist_ok=True)
    except Exception as e:
        rep.add("pipeline/out can be created", False, str(e))
    # Windows can transiently deny a create right after another process touched the
    # directory (Defender/indexer), so retry a few times before declaring failure.
    write_error = None
    for attempt in range(5):
        probe_file = os.path.join(OUT, f".doctor_write_test_{os.getpid()}")
        try:
            with open(probe_file, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe_file)
            write_error = None
            break
        except Exception as e:
            write_error = e
            time.sleep(0.3 * (attempt + 1))
    if write_error is None:
        rep.add("pipeline/out is writable", True, OUT)
    else:
        rep.add("pipeline/out is writable", False, f"{write_error}")

    try:
        usage = shutil.disk_usage(ROOT)
        free_gb = usage.free / (1024 ** 3)
        rep.add("free disk space", free_gb >= 5.0, f"{free_gb:.1f} GB free", "warn" if free_gb >= 2 else "fail")
    except Exception:
        pass

    # ----- optional hardware/model -----
    if shutil.which("nvidia-smi"):
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30, errors="replace")
            gpus = [x.strip() for x in (p.stdout or "").splitlines() if x.strip()]
            rep.add("CUDA GPU visible", bool(gpus), "; ".join(gpus) if gpus else "nvidia-smi returned nothing", "warn")
        except Exception as e:
            rep.add("CUDA GPU visible", False, str(e), "warn")
    else:
        rep.add("CUDA GPU visible", False, "nvidia-smi not found: runs will be CPU-only or fail", "warn")

    check_model(rep, args.model, profile)

    report = {
        "root": ROOT,
        "profile": args.profile,
        "model": args.model,
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

    print(f"\ndoctor: {rep.fails} failed, {rep.warns} warnings")
    if rep.fails:
        print("Fix the FAIL lines first. docs/SETUP.md has the setup steps.")
    return 0 if rep.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
