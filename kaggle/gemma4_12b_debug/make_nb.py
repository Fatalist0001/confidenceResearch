"""Debug kernel: gemma-4-12B adequacy on the CURRENT upstream llama.cpp.

No hs build here - we only ask llama-completion 8 easy questions and look at the
answers. Configs: GGUF chat template (thinking off, 1 and 2 GPUs), thinking on,
hand-made Gemma 4 prompt without/with the empty thought channel, and a raw
non-chat prompt. Builds llama.cpp at origin/master, target llama-completion.
"""

import json
from pathlib import Path

KERNEL_ID = "odinodinodin/hs-gemma4-12b-debug"


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
        print("STDERR:", p.stderr[-3000:])
        if check:
            raise SystemExit(f"command failed: {cmd}")
    return p

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv",
                     shell=True, text=True, capture_output=True).stdout)
print("nproc:", os.cpu_count())
print(subprocess.run("df -h /kaggle/working /tmp | tail -3", shell=True, text=True,
                     capture_output=True).stdout)
print("cmake:", subprocess.run("cmake --version | head -1", shell=True, text=True,
                               capture_output=True).stdout.strip())

WORK  = "/tmp/llama.cpp"
MODEL = "/tmp/models/gemma-4-12b-it-Q4_K_M.gguf"
BIN   = f"{WORK}/build/bin/llama-completion"
'''

BUILD = r'''
# current upstream, no pin
sh(f"rm -rf {WORK} && mkdir -p {WORK} && cd {WORK} && "
   "git init -q && git remote add origin https://github.com/ggml-org/llama.cpp && "
   "git fetch -q --depth 1 origin master && git checkout -q FETCH_HEAD")
print("llama.cpp:", sh(f"cd {WORK} && git log -1 --format='%h %ad %s' --date=short",
                       quiet=True).stdout.strip())

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
        print("WARNING: libcuda driver library not found")

BASE_ARGS = ("-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF "
             "-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF "
             "-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON "
             f"-DCMAKE_LIBRARY_PATH={STUBS} -DCUDAToolkit_ROOT=/usr/local/cuda")
cfg = sh(f"cd {WORK} && cmake -B build -DGGML_CUDA=ON {BASE_ARGS}", check=False, quiet=True)
if cfg.returncode != 0:
    print("configure FAILED, retrying with GGML_CUDA_NO_VMM=ON")
    sh(f"rm -rf {WORK}/build")
    sh(f"cd {WORK} && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON {BASE_ARGS}", quiet=True)
t0 = time.time()
sh(f"cd {WORK} && cmake --build build -j{os.cpu_count()} --target llama-completion", quiet=True)
print(f"BUILD DONE in {time.time()-t0:.1f}s")
print(subprocess.run(f"ls -la {WORK}/build/bin/ | head", shell=True, text=True,
                     capture_output=True).stdout)
'''

DOWNLOAD = r'''
from huggingface_hub import hf_hub_download
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

TESTS = r'''
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

# hand-made Gemma 4 prompts: same content, the only difference is the empty
# thought channel that the 12B GGUF template inserts when thinking is off
TPL_NOCH  = r"{{- '<|turn>user\n' + messages[0]['content'] + '<turn|>\n<|turn>model\n' -}}"
TPL_CH    = r"{{- '<|turn>user\n' + messages[0]['content'] + '<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' -}}"

# name, devices, n_predict, extra argv
CONFIGS = [
    ("tmpl_off_1gpu", "0",   32,  ["--jinja", "--chat-template-kwargs", '{"enable_thinking": false}']),
    ("tmpl_off_2gpu", "0,1", 32,  ["--jinja", "--chat-template-kwargs", '{"enable_thinking": false}']),
    ("tmpl_on_1gpu",  "0",   256, ["--jinja", "--chat-template-kwargs", '{"enable_thinking": true}']),
    ("hand_noch",     "0",   32,  ["--jinja", "--chat-template", TPL_NOCH]),
    ("hand_ch",       "0",   32,  ["--jinja", "--chat-template", TPL_CH]),
    ("raw_1gpu",      "0",   32,  ["-no-cnv"]),
]

DROP = re.compile(r"^(llama_|load time|prompt eval time|eval time|total time|sampling time|"
                  r"generate time|print_info|common_init|init:|$)", re.I)

def clean(text):
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if DROP.match(s):
            continue
        if " time =" in line or "tokens per second" in line:
            continue
        keep.append(s)
    return "\n".join(keep).strip()

results = {}
for name, dev, npred, extra in CONFIGS:
    print("=" * 78)
    print(f"### {name}: CUDA_VISIBLE_DEVICES={dev} n_predict={npred}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = dev
    env["LLAMA_LOG_LEVEL"] = "WARN"
    rows = {}
    for qid, q in QS:
        if "-no-cnv" in extra:
            prompt = q + "\nAnswer:"
        else:
            prompt = q + TAIL
        cmd = [BIN, "-m", MODEL, "-p", prompt, "-n", str(npred), "--temp", "0",
               "-ngl", "99", "-st", "--no-warmup", "--no-display-prompt",
               "-c", "1024", "-t", "4", "--seed", "0"] + extra
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = -9, (e.stdout or b"").decode("utf-8", "replace"), "TIMEOUT"
        dt = time.time() - t0
        ans = clean(out)
        rows[qid] = {"q": q, "answer": ans, "secs": round(dt, 1), "rc": rc}
        print(f"[{qid:9s}] {dt:5.1f}s rc={rc} -> {ans[:220]!r}")
        if rc != 0:
            print("   STDERR:", err[-400:])
    results[name] = rows
    print()

EXPECT = {"hamlet": "shakespeare", "paris": "paris", "math": "51", "monalisa": "leonardo",
          "mars": "mars", "gold": "au", "ballo": "verdi", "kipling": "kipling"}
print("=" * 78)
for name, rows in results.items():
    ok = sum(1 for k, v in rows.items() if EXPECT[k] in v["answer"].lower())
    print(f"{name:14s} correct {ok}/8")
    for k in EXPECT:
        hit = "ok " if EXPECT[k] in rows[k]["answer"].lower() else "   "
        print(f"   {hit}{k:9s} {rows[k]['answer'][:140]!r}")

json.dump({"model": "gemma-4-12b-it-Q4_K_M",
           "llama_cpp": subprocess.run(f"cd {WORK} && git log -1 --format='%h %ad %s' --date=short",
                                       shell=True, text=True, capture_output=True).stdout.strip(),
           "configs": [c[0] for c in CONFIGS],
           "results": results},
          open("/kaggle/working/answers.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("wrote /kaggle/working/answers.json")
'''

PROMPT = r'''
# what exactly does llama-completion render? one question, thinking off, verbose prompt
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
cmd = [BIN, "-m", MODEL, "-p", QS[0][1] + TAIL, "-n", "8", "--temp", "0", "-ngl", "99",
       "--jinja", "--chat-template-kwargs", '{"enable_thinking": false}',
       "-st", "--no-warmup", "--verbose-prompt", "-c", "1024", "-t", "4", "--seed", "0"]
p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
print("rc:", p.returncode)
print(repr(p.stdout[:1500]))
print("----- stderr (prompt dump) -----")
print(p.stderr[-2500:])
'''

cells = [
    code("setup", SETUP),
    code("build", BUILD),
    code("download", DOWNLOAD),
    code("tests", TESTS),
    code("prompt", PROMPT),
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

with open(OUTDIR / "g12dbg.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

meta = {
    "id": KERNEL_ID,
    "title": "hs gemma4 12b debug",
    "code_file": "g12dbg.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}
with open(OUTDIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("wrote", len(cells), "cells to", OUTDIR)