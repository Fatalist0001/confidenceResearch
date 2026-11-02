"""Gate: our hs built on current master, gemma-4-12B answers must be sane.

The pin (d4cff114c) produces meta-text on 12B both on one and on two GPUs, while
the same model answers 8/8 on master. This kernel rebuilds the tool at master and
checks that the answers are real answers before any full run:
  - 8 easy questions (expect 8/8),
  - the real pilot 200 rows: count truncated / meta-text answers, show samples.
Only --layers 0 is captured here, so the hidden arrays stay tiny; this run is
about the answers and about speed.
"""

import json
from pathlib import Path

KERNEL_ID = "odinodinodin/hs-gemma4-12b-master-gate"


def code(src_id, src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": src_id,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


SETUP = r'''
import os, subprocess, time, json, re, sys, glob, shutil

def sh(cmd, check=True, quiet=False):
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    dt = time.time() - t0
    if not quiet:
        print(f"$ {cmd}\n  -> rc={p.returncode}  {dt:.1f}s")
    if p.stdout.strip():
        print(p.stdout[-2500:])
    if p.returncode != 0:
        print("STDERR:", p.stderr[-2500:])
        if check:
            raise SystemExit(f"command failed: {cmd}")
    return p

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True,
                     text=True, capture_output=True).stdout)
print("nproc:", os.cpu_count())

WORK  = "/tmp/llama.cpp"
MODEL = "/tmp/models/gemma-4-12b-it-Q4_K_M.gguf"
BIN   = f"{WORK}/build/bin/llama-hidden-states"
'''

BUILD = r'''
found = {}
for root, dirs, files in os.walk("/kaggle/input"):
    for fn in files:
        if fn in ("hidden-states.cpp", "hidden-states-CMakeLists.txt", "examples-CMakeLists.txt"):
            found.setdefault(fn, os.path.join(root, fn))
print("dataset files:", found)
assert "hidden-states.cpp" in found, "hs-src dataset not mounted"
src_txt = open(found["hidden-states.cpp"], encoding="utf-8").read()
print("ported source:", 'res.prompt.str()' in src_txt)

sh(f"rm -rf {WORK} && mkdir -p {WORK} && cd {WORK} && "
   "git init -q && git remote add origin https://github.com/ggml-org/llama.cpp && "
   "git fetch -q --depth 1 origin master && git checkout -q FETCH_HEAD")
print("llama.cpp:", sh(f"cd {WORK} && git log -1 --format='%h %ad %s' --date=short",
                       quiet=True).stdout.strip())

os.makedirs(WORK + "/examples/hidden-states", exist_ok=True)
shutil.copyfile(found["hidden-states.cpp"], WORK + "/examples/hidden-states/hidden-states.cpp")
shutil.copyfile(found["hidden-states-CMakeLists.txt"], WORK + "/examples/hidden-states/CMakeLists.txt")
shutil.copyfile(found["examples-CMakeLists.txt"], WORK + "/examples/CMakeLists.txt")

STUBS = "/tmp/cuda_stubs"
os.makedirs(STUBS, exist_ok=True)
if not os.path.exists(STUBS + "/libcuda.so"):
    for cand in ("/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1",
                 "/usr/local/cuda/compat/libcuda.so.1",
                 "/usr/local/cuda/lib64/stubs/libcuda.so",
                 "/usr/local/cuda/compat/libcuda.so"):
        if not os.path.exists(cand):
            continue
        try:
            os.symlink(cand, STUBS + "/libcuda.so")
        except OSError:
            shutil.copyfile(cand, STUBS + "/libcuda.so")
        print("stub:", cand)
        break

BASE_ARGS = ("-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF "
             "-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF "
             "-DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=ON "
             f"-DCMAKE_LIBRARY_PATH={STUBS} -DCUDAToolkit_ROOT=/usr/local/cuda")
cfg = sh(f"cd {WORK} && cmake -B build -DGGML_CUDA=ON {BASE_ARGS}", check=False, quiet=True)
if cfg.returncode != 0:
    print("configure FAILED, retry with GGML_CUDA_NO_VMM=ON")
    sh(f"rm -rf {WORK}/build")
    sh(f"cd {WORK} && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON {BASE_ARGS}", quiet=True)
t0 = time.time()
sh(f"cd {WORK} && cmake --build build -j{os.cpu_count()} --target llama-hidden-states", quiet=True)
print(f"BUILD DONE in {time.time()-t0:.1f}s")
'''

DATA = r'''
from huggingface_hub import hf_hub_download
os.makedirs("/tmp/models", exist_ok=True)
path = hf_hub_download(repo_id="unsloth/gemma-4-12b-it-GGUF",
                       filename="gemma-4-12b-it-Q4_K_M.gguf", local_dir="/tmp/models")
if path != MODEL:
    os.replace(path, MODEL)
print("model:", MODEL, f"{os.path.getsize(MODEL)/2**30:.2f} GiB")

QS = [
    ("hamlet",   "Who wrote the play Hamlet?"),
    ("paris",    "What is the capital of France?"),
    ("math",     "What is 17 times 3?"),
    ("monalisa", "Who painted the Mona Lisa?"),
    ("mars",     "Which planet is known as the Red Planet?"),
    ("gold",     "What is the chemical symbol for gold?"),
    ("ballo",    "Who wrote the opera A Masked Ball?"),
    ("kipling",  "Which novelist, born in Bombay in 1865, won the Nobel Prize in Literature in 1907?"),
]
TAIL = " Answer with just the answer, no explanation."
with open("/tmp/easy8.jsonl", "w", encoding="utf-8") as f:
    for qid, q in QS:
        f.write(json.dumps({"id": qid, "question": q + TAIL}, ensure_ascii=False) + "\n")

PILOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "pilot.jsonl" in files:
        PILOT = os.path.join(root, "pilot.jsonl")
print("pilot:", PILOT, sum(1 for _ in open(PILOT, encoding="utf-8")))
'''

RUN = r'''
EXPECT = {"hamlet": "shakespeare", "paris": "paris", "math": "51", "monalisa": "leonardo",
          "mars": "mars", "gold": "au", "ballo": "verdi", "kipling": "kipling"}
META_PAT = re.compile(r"the answer is the answer|the correct answer is|the prompt asks|"
                      r"the result of the calculation is 1|the answer, no explanation", re.I)

def run_hs(name, batch, n_predict):
    out = f"/kaggle/working/out/{name}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = (f"{BIN} -m {MODEL} --batch {batch} --generate --max-new-tokens {n_predict} "
           f"--max-len 1024 --layers 0 -ngl 99 -sm none --out {out}")
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env, timeout=5400)
    dt = time.time() - t0
    print(f"### {name}: rc={p.returncode}  {dt:.1f}s")
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print("STDERR:", p.stderr[-2000:])
    rows = []
    meta = os.path.join(out, "meta.jsonl")
    if os.path.exists(meta):
        rows = [json.loads(l) for l in open(meta, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    trunc = [r for r in ok if r.get("truncated")]
    meta_txt = [r for r in ok if META_PAT.search(r.get("answer") or "")]
    print(f"    rows={len(rows)} ok={len(ok)} truncated={len(trunc)} "
          f"meta_text={len(meta_txt)} secs={dt:.1f} ex/s={len(rows)/dt:.2f}")
    return rows, dt

rows8, dt8 = run_hs("easy8", "/tmp/easy8.jsonl", 32)
print("--- easy8 answers ---")
n_ok = 0
for r in rows8:
    hit = EXPECT.get(r["id"], "\0") in (r.get("answer") or "").lower()
    n_ok += int(hit)
    print(f"   {'ok ' if hit else '   '}{r['id']:9s} -> {(r.get('answer') or '')[:120]!r}")
print(f"easy8 correct {n_ok}/{len(rows8)}")

rows200, dt200 = run_hs("pilot200", PILOT, 64)
print("--- pilot200 samples (12) ---")
print(f"    time {dt200:.1f}s -> full 6000 estimate {6000/(len(rows200)/dt200)/60:.1f} min")
for r in rows200[:12]:
    print(f"   {r['id']:10s} trunc={int(bool(r.get('truncated')))} "
          f"n={r.get('n_tokens'):3d} -> {(r.get('answer') or '')[:130]!r}")

json.dump({"llama_cpp": sh(f"cd {WORK} && git log -1 --format='%h %ad %s' --date=short",
                           quiet=True).stdout.strip(),
           "easy8_correct": n_ok, "easy8": {r["id"]: r.get("answer") for r in rows8},
           "pilot200_rows": len(rows200),
           "pilot200_seconds": round(dt200, 1),
           "pilot200_meta_text": sum(1 for r in rows200 if META_PAT.search(r.get("answer") or "")),
           "pilot200_truncated": sum(1 for r in rows200 if r.get("truncated")),
           "pilot200": {r["id"]: r.get("answer") for r in rows200}},
          open("/kaggle/working/gate.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote /kaggle/working/gate.json")
shutil.copyfile(os.path.join("/kaggle/working/out/pilot200", "meta.jsonl"),
                "/kaggle/working/pilot200_meta.jsonl")
print("meta size:", os.path.getsize("/kaggle/working/pilot200_meta.jsonl")/1e6, "MB")
'''

cells = [
    code("setup", SETUP),
    code("build", BUILD),
    code("data", DATA),
    code("run", RUN),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTDIR = Path(__file__).resolve().parent / "_build" / "kernel"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(OUTDIR / "g12gate.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

meta = {
    "id": KERNEL_ID,
    "title": "hs gemma4 12b master gate",
    "code_file": "g12gate.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["odinodinodin/hs-src"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}
with open(OUTDIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("wrote", len(cells), "cells to", OUTDIR)