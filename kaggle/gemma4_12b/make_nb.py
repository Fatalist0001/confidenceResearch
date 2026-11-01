import json


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
import os, subprocess, time, json, re, sys, textwrap

def sh(cmd, check=True, quiet=False):
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    dt = time.time() - t0
    if not quiet:
        print(f"$ {cmd}\n  -> rc={p.returncode}  {dt:.1f}s")
    if p.stdout.strip():
        print(p.stdout[-3000:])
    if p.returncode != 0:
        print("STDERR:", p.stderr[-4000:])
        if check:
            raise SystemExit(f"command failed: {cmd}")
    return p

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True,
                     text=True, capture_output=True).stdout)
print("nproc:", os.cpu_count())
print(subprocess.run("df -h /kaggle/working /tmp | tail -3", shell=True, text=True,
                     capture_output=True).stdout)

print("=== /kaggle/input ===")
print(subprocess.run("ls -la /kaggle/input", shell=True, text=True, capture_output=True).stdout)
print(subprocess.run("find /kaggle/input -maxdepth 3 | head -40", shell=True, text=True, capture_output=True).stdout)

SRC = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "hidden-states.cpp" in files:
        SRC = root
        break
if SRC is None:
    raise SystemExit("FATAL: hidden-states.cpp not found under /kaggle/input")
print("SRC =", SRC)

WORK  = "/tmp/llama.cpp"
MODEL = "/tmp/models/gemma-4-12b-it-Q4_K_M.gguf"
BIN   = f"{WORK}/build/bin/llama-hidden-states"
OUT_PILOT = "/kaggle/working/out/pilot_gemma4_12b"
OUT_MAIN  = "/kaggle/working/out/main_gemma4_12b"
N_PILOT   = 40
print("src listing:", subprocess.run(f"find {SRC} -maxdepth 4 -type f | head -20",
                                     shell=True, text=True, capture_output=True).stdout)
'''

BUILD = r'''
COMMIT = "d4cff114c0084f1fbc9b4c62717eca8fb2ae494a"
sh(f"rm -rf {WORK} && mkdir -p {WORK} && cd {WORK} && "
   "git init -q && git remote add origin https://github.com/ggml-org/llama.cpp && "
   f"git fetch -q --depth 1 origin {COMMIT} && git checkout -q FETCH_HEAD")
sh(f"cd {WORK} && git log --oneline -1")
# our two files + the one-line examples/CMakeLists change
sh(f"mkdir -p {WORK}/examples/hidden-states && "
   f"cp {SRC}/hidden-states.cpp {WORK}/examples/hidden-states/ && "
   f"cp {SRC}/hidden-states-CMakeLists.txt {WORK}/examples/hidden-states/CMakeLists.txt && "
   f"cp {SRC}/examples-CMakeLists.txt {WORK}/examples/CMakeLists.txt")
print(subprocess.run("ls -la /usr/local/cuda/lib64/stubs 2>&1 | head -8; echo ---; "
                     "ls -la /usr/local/cuda/compat 2>&1 | head -8; echo ---; "
                     "find / -maxdepth 6 -name 'libcuda.so*' 2>/dev/null | head -10",
                     shell=True, text=True, capture_output=True).stdout)

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
        except OSError as e:
            print("symlink failed:", e, "-> copying")
            import shutil
            shutil.copyfile(cand, STUBS + "/libcuda.so")
        print("stub:", cand, "->", STUBS + "/libcuda.so")
        break
    else:
        print("WARNING: libcuda driver library not found anywhere")

BASE_ARGS = ("-DCMAKE_BUILD_TYPE=Release "
             "-DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF "
             "-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF "
             "-DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=OFF "
             f"-DCMAKE_LIBRARY_PATH={STUBS} -DCUDAToolkit_ROOT=/usr/local/cuda")
cfg = sh(f"cd {WORK} && cmake -B build -DGGML_CUDA=ON {BASE_ARGS}", check=False, quiet=True)
if cfg.returncode != 0:
    print("configure FAILED, retrying with GGML_CUDA_NO_VMM=ON (no libcuda dev stub)")
    sh(f"rm -rf {WORK}/build")
    sh(f"cd {WORK} && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON {BASE_ARGS}", quiet=True)
t0 = time.time()
sh(f"cd {WORK} && cmake --build build -j{os.cpu_count()} --target llama-hidden-states", quiet=True)
print(f"BUILD DONE in {time.time()-t0:.1f}s")
print(subprocess.run(f"ls -la {WORK}/build/bin/", shell=True, text=True,
                     capture_output=True).stdout)
'''

DOWNLOAD = r'''
from huggingface_hub import hf_hub_download
import os, time
os.makedirs("/tmp/models", exist_ok=True)
t0 = time.time()
path = hf_hub_download(repo_id="unsloth/gemma-4-12b-it-GGUF",
                       filename="gemma-4-12b-it-Q4_K_M.gguf",
                       local_dir="/tmp/models")
print("downloaded:", path, f"{os.path.getsize(path)/2**30:.2f} GiB", f"in {time.time()-t0:.0f}s")
if path != MODEL:
    os.replace(path, MODEL)
print("model at", MODEL, f"{os.path.getsize(MODEL)/2**30:.2f} GiB")
'''

PILOT = r'''
sh(f"head -{N_PILOT} {SRC}/pilot.jsonl > /kaggle/working/pilot40.jsonl && wc -l /kaggle/working/pilot40.jsonl")
cmd = (f"{BIN} -m {MODEL} --batch /kaggle/working/pilot40.jsonl "
       "--generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99 "
       f"--out {OUT_PILOT} > /kaggle/working/pilot.log 2>&1")
t0 = time.time()
p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
print("rc:", p.returncode, f"wall {time.time()-t0:.0f}s")
print(open("/kaggle/working/pilot.log", errors="replace").read()[-3000:])
'''

GATE = r'''
log = open("/kaggle/working/pilot.log", errors="replace").read()
m = re.search(r"done:\s*(\d+)/(\d+)\s+ok=(\d+)\s+skipped=(\d+)\s+truncated=(\d+)\s+in\s+([\d.]+)\s*s\s*\(([\d.]+)\s*ex/s\)", log)
shp = re.search(r"shape=\[(\d+),\s*(\d+),\s*(\d+)\]", log)
print("summary match:", m.groups() if m else None)
print("shape match:", shp.groups() if shp else None)
filtered = re.findall(r"^\d+\.\d+\.\d+ I\s+(\S+)", log, flags=re.M)
print("non-l_out chans (sample):", sorted(set(filtered))[:5])

if not m or not shp:
    print("GATE FAIL: no summary in pilot log -> not running full set")
    ok_run = False
else:
    done, total, ok, skipped, trunc, secs, exs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6)), float(m.group(7))
    n, layers, dim = map(int, shp.groups())
    est_h = 6000.0 / exs / 3600.0
    print(f"ok={ok}/{total} skipped={skipped} truncated={trunc} ex/s={exs:.2f} "
          f"shape=[{n},{layers},{dim}] est_full={est_h*60:.0f} min")
    ok_run = (ok == total) and skipped == 0 and exs > 0.2 and est_h < 4.0
    print("GATE", "PASS" if ok_run else "FAIL")
'''

MAIN = r'''
if ok_run:
    cmd = (f"{BIN} -m {MODEL} --batch {SRC}/main.jsonl "
           "--generate --max-new-tokens 64 --max-len 1024 --layers all -ngl 99 "
           f"--out {OUT_MAIN} > /kaggle/working/main_gemma4_12b.log 2>&1")
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print("FULL rc:", p.returncode, f"wall {time.time()-t0:.0f}s")
    print(open("/kaggle/working/main_gemma4_12b.log", errors="replace").read()[-1500:])
else:
    print("skipped full run by gate")
'''

FINAL = r'''
import glob
rep = {}
for name, d in (("pilot", OUT_PILOT), ("main", OUT_MAIN)):
    if not os.path.isdir(d):
        continue
    files = {}
    for f in sorted(glob.glob(d + "/*")):
        files[os.path.basename(f)] = round(os.path.getsize(f) / 2**20, 1)
    meta = d + "/meta.jsonl"
    info = {"files_MB": files}
    if os.path.exists(meta):
        rows = [json.loads(l) for l in open(meta, encoding="utf-8")]
        info["meta_rows"] = len(rows)
        info["status_ok"] = sum(1 for r in rows if r.get("status") == "ok")
        info["truncated"] = sum(1 for r in rows if r.get("truncated"))
        info["ids_unique"] = len({r.get("id") for r in rows})
    cfg = d + "/config.json"
    if os.path.exists(cfg):
        c = json.load(open(cfg, encoding="utf-8"))
        info["config"] = {k: c.get(k) for k in ("n_layers", "n_embd", "layers_requested", "max_len", "template", "thinking", "n_examples")}
    rep[name] = info
print(json.dumps(rep, ensure_ascii=False, indent=2))
json.dump(rep, open("/kaggle/working/run_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(subprocess.run("du -sh /kaggle/working/out /kaggle/working/*.log 2>/dev/null", shell=True, text=True, capture_output=True).stdout)
'''

cells = [
    code("setup", SETUP),
    code("build", BUILD),
    code("download", DOWNLOAD),
    code("pilot", PILOT),
    code("gate", GATE),
    code("main", MAIN),
    code("final", FINAL),
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

import os
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "_build" / "kernel"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(OUTDIR / "hs12b.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

meta = {
    "id": "odinodinodin/hs-gemma4-12b",
    "title": "hs gemma4 12b",
    "code_file": "hs12b.ipynb",
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

print("wrote notebook with", len(cells), "cells to", OUTDIR)
