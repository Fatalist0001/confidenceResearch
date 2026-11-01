"""Pin build of llama-hidden-states on 1 vs 2 GPUs (gemma-4-12B).

The rendered prompt of the failed 12B run was already canonical (it matches the
hand template that answers 8/8 on current master), so the prompt is not the
problem. This kernel keeps the pin and only varies the GPU split:
  1gpu  - CUDA_VISIBLE_DEVICES=0,   -sm none
  2gpu  - CUDA_VISIBLE_DEVICES=0,1, -sm layer
8 easy questions + the real pilot 40 rows, answers compared side by side.
"""

import json
from pathlib import Path

KERNEL_ID = "odinodinodin/hs-gemma4-12b-pin-gpu"


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
COMMIT = "d4cff114c0084f1fbc9b4c62717eca8fb2ae494a"   # the pin our runs use
'''

BUILD = r'''
# --- our sources from the hs-src dataset (flat layout, find by name) ---
found = {}
for root, dirs, files in os.walk("/kaggle/input"):
    for fn in files:
        if fn in ("hidden-states.cpp", "hidden-states-CMakeLists.txt", "examples-CMakeLists.txt"):
            found.setdefault(fn, os.path.join(root, fn))
print("dataset files:", found)
assert "hidden-states.cpp" in found, "hs-src dataset not mounted"

sh(f"rm -rf {WORK} && mkdir -p {WORK} && cd {WORK} && "
   "git init -q && git remote add origin https://github.com/ggml-org/llama.cpp && "
   f"git fetch -q --depth 1 origin {COMMIT} && git checkout -q FETCH_HEAD")
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
print(subprocess.run(f"ls -la {WORK}/build/bin/ | head", shell=True, text=True,
                     capture_output=True).stdout)
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
print(open("/tmp/easy8.jsonl", encoding="utf-8").read().splitlines()[0])

PILOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "pilot.jsonl" in files:
        PILOT = os.path.join(root, "pilot.jsonl")
print("pilot:", PILOT)
'''

RUN = r'''
EXPECT = {"hamlet": "shakespeare", "paris": "paris", "math": "51", "monalisa": "leonardo",
          "mars": "mars", "gold": "au", "ballo": "verdi", "kipling": "kipling"}

CONFIGS = [
    ("pin_1gpu", "0",   ["-sm", "none"]),
    ("pin_2gpu", "0,1", ["-sm", "layer"]),
]

summary = {}
for name, dev, extra in CONFIGS:
    out = f"/kaggle/working/out/{name}"
    print("=" * 78)
    print(f"### {name}: CUDA_VISIBLE_DEVICES={dev} {extra}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = dev
    cmd = (f"{BIN} -m {MODEL} --batch /tmp/easy8.jsonl --generate --max-new-tokens 32 "
           f"--max-len 1024 --layers all -ngl 99 {' '.join(extra)} --out {out}")
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env, timeout=3600)
    print(f"rc={p.returncode}  {time.time()-t0:.1f}s")
    print(p.stdout[-1200:])
    if p.returncode != 0:
        print("STDERR:", p.stderr[-2500:])
    rows = []
    meta = os.path.join(out, "meta.jsonl")
    if os.path.exists(meta):
        rows = [json.loads(l) for l in open(meta, encoding="utf-8") if l.strip()]
    ok = sum(1 for r in rows if EXPECT.get(r["id"], "\0") in (r.get("answer") or "").lower())
    print(f"correct {ok}/{len(rows)}")
    for r in rows:
        print(f"   {r['id']:9s} trunc={int(bool(r.get('truncated')))} "
              f"n={r.get('n_tokens')} -> {(r.get('answer') or '')[:150]!r}")
    summary[name] = {"rc": p.returncode, "rows": len(rows), "correct": ok,
                     "answers": {r["id"]: r.get("answer") for r in rows},
                     "n_tokens": {r["id"]: r.get("n_tokens") for r in rows},
                     "truncated": {r["id"]: r.get("truncated") for r in rows}}
    print()

# real pilot rows on the single-GPU config, for a like-for-like view
if PILOT:
    out = "/kaggle/working/out/pin_1gpu_pilot40"
    print("=" * 78)
    print("### pin_1gpu, pilot 40")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = (f"{BIN} -m {MODEL} --batch {PILOT} --generate --max-new-tokens 64 "
           f"--max-len 1024 --layers all -ngl 99 -sm none --out {out}")
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env, timeout=3600)
    print(f"rc={p.returncode}  {time.time()-t0:.1f}s")
    print(p.stdout[-1200:])
    rows = [json.loads(l) for l in open(os.path.join(out, "meta.jsonl"), encoding="utf-8") if l.strip()]
    print(f"rows={len(rows)} truncated={sum(1 for r in rows if r.get('truncated'))}")
    for r in rows[:12]:
        print(f"   {r['id']:9s} -> {(r.get('answer') or '')[:140]!r}")
    summary["pilot40_1gpu"] = {"rows": len(rows),
                               "truncated": sum(1 for r in rows if r.get("truncated")),
                               "answers": {r["id"]: r.get("answer") for r in rows}}

json.dump({"llama_cpp": sh(f"cd {WORK} && git log -1 --format='%h %ad %s' --date=short",
                           quiet=True).stdout.strip(),
           "summary": summary},
          open("/kaggle/working/pin_gpu.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote /kaggle/working/pin_gpu.json")
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

with open(OUTDIR / "g12pin.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

meta = {
    "id": KERNEL_ID,
    "title": "hs gemma4 12b pin 1gpu 2gpu",
    "code_file": "g12pin.ipynb",
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