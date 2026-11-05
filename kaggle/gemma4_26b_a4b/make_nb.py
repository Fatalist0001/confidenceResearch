import json
from pathlib import Path


KERNEL_ID = "odinodinodin/hs-gemma4-26b-a4b"


def code(cell_id, source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


SETUP = r'''
import collections
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time

TAG = "gemma4_26b_a4b_udq4km"
WORK = "/tmp/llama.cpp"
MODEL_DIR = "/tmp/models"
MODEL = MODEL_DIR + "/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
BIN = WORK + "/build/bin/llama-hidden-states"
DUMP = WORK + "/build/bin/hs-prompt-dump"
OUT_ROOT = "/kaggle/working/out"
OUT_EASY = OUT_ROOT + "/easy8"
OUT_PILOT = OUT_ROOT + "/pilot_" + TAG
OUT_MAIN = OUT_ROOT + "/main_" + TAG
PILOT_INPUT = "/kaggle/working/pilot40.jsonl"
MAIN_LOG = "/kaggle/working/main_" + TAG + ".log"

LLAMA_COMMIT = "785322d17f2d51e5185cf996a1f4e46d87640841"
LLAMA_SHORT = "785322d17"
LLAMA_BUILD = 12033
HF_REPO = "unsloth/gemma-4-26B-A4B-it-GGUF"
HF_FILE = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
HF_REVISION = "c099eb48e663fd284577b04978a94ffccb261841"
MODEL_BYTES = 16947541728
MODEL_SHA256 = "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f"
EXPECTED_LAYERS = 30
EXPECTED_EMBD = 2816
N_PILOT = 40
N_MAIN = 6000
INSTRUCTION = "Answer with just the answer, no explanation."
GPU_ARGS = ["-ngl", "99", "-sm", "layer", "-ts", "1,1"]


def write_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sh(command, check=True, log=None):
    t0 = time.time()
    if log:
        with open(log, "w", encoding="utf-8") as out:
            proc = subprocess.run(command, shell=True, text=True, stdout=out, stderr=subprocess.STDOUT)
        tail = open(log, encoding="utf-8", errors="replace").read()[-4000:]
    else:
        proc = subprocess.run(command, shell=True, text=True, capture_output=True)
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:]
    print(f"$ {command}\n  -> rc={proc.returncode} in {time.time()-t0:.1f}s")
    if tail.strip():
        print(tail)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {command}")
    return proc


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def gpu_snapshot():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    values = []
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                try:
                    values.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass
    return values


def run_hs(name, batch, out_dir, layers, max_new_tokens, timeout_s):
    log = f"/kaggle/working/{name}.log"
    args = [
        BIN, "-m", MODEL, "--batch", batch, "--generate",
        "--max-new-tokens", str(max_new_tokens), "--max-len", "1024",
        "--layers", layers, "--out", out_dir,
    ] + GPU_ARGS
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    print("RUN:", " ".join(args))
    peaks_mem = [0, 0]
    peaks_util = [0, 0]
    t0 = time.time()
    timed_out = False
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            for i, (mem, util) in enumerate(gpu_snapshot()[:2]):
                peaks_mem[i] = max(peaks_mem[i], mem)
                peaks_util[i] = max(peaks_util[i], util)
            if time.time() - t0 > timeout_s:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(0.5)
        rc = proc.wait()
    elapsed = time.time() - t0
    text = open(log, encoding="utf-8", errors="replace").read()
    print(f"{name}: rc={rc} elapsed={elapsed:.1f}s timeout={timed_out} "
          f"peak_mem_MiB={peaks_mem} peak_util={peaks_util}")
    print(text[-3500:])
    return {
        "name": name,
        "rc": rc,
        "elapsed_s": round(elapsed, 1),
        "timed_out": timed_out,
        "peak_mem_MiB": peaks_mem,
        "peak_util_pct": peaks_util,
        "log": log,
    }


os.makedirs(OUT_ROOT, exist_ok=True)
print(subprocess.run(
    "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    shell=True, text=True, capture_output=True,
).stdout)
gpu_info = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
    text=True, capture_output=True,
).stdout.splitlines()
if len(gpu_info) < 2:
    write_json("/kaggle/working/preflight_failure.json", {"error": "need two GPUs", "gpu_info": gpu_info})
    raise RuntimeError(f"expected two T4 GPUs, got: {gpu_info}")
gpu_mem = [int(line.rsplit(",", 1)[1].strip()) for line in gpu_info[:2]]
if min(gpu_mem) < 14000:
    write_json("/kaggle/working/preflight_failure.json", {"error": "GPU memory too small", "gpu_info": gpu_info})
    raise RuntimeError(f"GPU memory too small: {gpu_info}")

print("cpu_count:", os.cpu_count())
print(subprocess.run("df -h /kaggle/working /tmp", shell=True, text=True, capture_output=True).stdout)
tmp_free = shutil.disk_usage("/tmp").free
working_free = shutil.disk_usage("/kaggle/working").free
if tmp_free < 20 * 2**30 or working_free < 5 * 2**30:
    write_json("/kaggle/working/preflight_failure.json", {
        "error": "insufficient disk before build/download",
        "tmp_free_GiB": tmp_free / 2**30,
        "working_free_GiB": working_free / 2**30,
    })
    raise RuntimeError("insufficient disk")

SRC = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "source-manifest.json" in files and "hidden-states.cpp" in files:
        SRC = root
        break
if SRC is None:
    raise RuntimeError("source-manifest.json and hidden-states.cpp not found under /kaggle/input")
manifest = json.load(open(os.path.join(SRC, "source-manifest.json"), encoding="utf-8"))
source_errors = []
if manifest != EXPECTED_SOURCE_MANIFEST:
    source_errors.append("mounted source-manifest.json differs from the kernel-pinned manifest")
for filename, expected in manifest["files"].items():
    path = os.path.join(SRC, filename)
    if not os.path.exists(path):
        source_errors.append(f"missing {filename}")
        continue
    got_size = os.path.getsize(path)
    got_hash = sha256_file(path)
    if got_size != expected["bytes"] or got_hash.lower() != expected["sha256"].lower():
        source_errors.append(f"{filename}: size/hash mismatch")
if source_errors:
    write_json("/kaggle/working/preflight_failure.json", {"error": "dataset hash mismatch", "details": source_errors})
    raise RuntimeError(str(source_errors))
print("dataset source verified:", SRC, manifest)

PILOT_SOURCE = os.path.join(SRC, "pilot.jsonl")
MAIN_SOURCE = os.path.join(SRC, "main.jsonl")
pilot_rows_all = read_jsonl(PILOT_SOURCE)
main_rows_all = read_jsonl(MAIN_SOURCE)
input_errors = []
if len(pilot_rows_all) != 200:
    input_errors.append(f"pilot rows {len(pilot_rows_all)} != 200")
if len(main_rows_all) != N_MAIN:
    input_errors.append(f"main rows {len(main_rows_all)} != {N_MAIN}")
for label, rows in (("pilot", pilot_rows_all), ("main", main_rows_all)):
    empty = sum(not str(row.get("question", "")).strip() for row in rows)
    bad_instruction = sum(str(row.get("question", "")).count(INSTRUCTION) != 1 for row in rows)
    if empty or bad_instruction:
        input_errors.append(f"{label}: empty={empty} bad_instruction={bad_instruction}")
dups = {key: value for key, value in collections.Counter(str(r.get("id")) for r in main_rows_all).items() if value > 1}
if dups != {"bb_956": 2}:
    input_errors.append(f"main duplicate ids: {dups}")
if input_errors:
    write_json("/kaggle/working/preflight_failure.json", {"error": "input validation", "details": input_errors})
    raise RuntimeError(str(input_errors))
with open(PILOT_SOURCE, encoding="utf-8") as src, open(PILOT_INPUT, "w", encoding="utf-8", newline="\n") as dst:
    for _ in range(N_PILOT):
        dst.write(next(src))
write_json("/kaggle/working/input_preflight.json", {
    "pilot_rows": len(pilot_rows_all),
    "main_rows": len(main_rows_all),
    "pilot40_sha256": sha256_file(PILOT_INPUT),
    "main_sha256": sha256_file(MAIN_SOURCE),
    "duplicate_ids": dups,
    "first_id": main_rows_all[0]["id"],
    "last_id": main_rows_all[-1]["id"],
})
print("input preflight passed")
'''


BUILD = r'''
BUILD_LOG = "/kaggle/working/build.log"
sh(f"rm -rf {WORK} && mkdir -p {WORK} && cd {WORK} && "
   "git init -q && git remote add origin https://github.com/ggml-org/llama.cpp && "
   f"git fetch -q --depth 1 origin {LLAMA_COMMIT} && git checkout -q FETCH_HEAD",
   log=BUILD_LOG)
checked_out = sh(f"cd {WORK} && git rev-parse HEAD", log=None).stdout.strip()
if checked_out != LLAMA_COMMIT:
    raise RuntimeError(f"wrong llama.cpp commit: {checked_out}")

os.makedirs(WORK + "/examples/hidden-states", exist_ok=True)
for src_name, dst_name in (
    ("hidden-states.cpp", "hidden-states.cpp"),
    ("hidden-states-CMakeLists.txt", "CMakeLists.txt"),
    ("prompt_dump.cpp", "prompt_dump.cpp"),
):
    shutil.copyfile(os.path.join(SRC, src_name), WORK + "/examples/hidden-states/" + dst_name)
shutil.copyfile(os.path.join(SRC, "examples-CMakeLists.txt"), WORK + "/examples/CMakeLists.txt")

STUBS = "/tmp/cuda_stubs"
os.makedirs(STUBS, exist_ok=True)
if not os.path.exists(STUBS + "/libcuda.so"):
    for candidate in (
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/local/cuda/compat/libcuda.so.1",
        "/usr/local/cuda/lib64/stubs/libcuda.so",
        "/usr/local/cuda/compat/libcuda.so",
    ):
        if os.path.exists(candidate):
            os.symlink(candidate, STUBS + "/libcuda.so")
            print("cuda driver stub:", candidate)
            break
    else:
        raise RuntimeError("libcuda.so not found")

BASE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON "
    "-DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=OFF "
    "-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF "
    "-DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=ON "
    f"-DLLAMA_BUILD_COMMIT={LLAMA_SHORT} -DLLAMA_BUILD_NUMBER={LLAMA_BUILD} "
    f"-DCMAKE_LIBRARY_PATH={STUBS} -DCUDAToolkit_ROOT=/usr/local/cuda"
)
cfg = sh(f"cd {WORK} && cmake -B build {BASE_ARGS}", check=False,
         log="/kaggle/working/configure.log")
if cfg.returncode != 0:
    sh(f"rm -rf {WORK}/build")
    sh(f"cd {WORK} && cmake -B build -DGGML_CUDA_NO_VMM=ON {BASE_ARGS}",
       log="/kaggle/working/configure.log")
sh(f"cd {WORK} && cmake --build build -j{os.cpu_count()} --target llama-hidden-states hs-prompt-dump",
   log=BUILD_LOG)
if not os.path.exists(BIN) or not os.path.exists(DUMP):
    raise RuntimeError("expected binaries are missing")
version_proc = subprocess.run([BIN, "--version"], text=True, capture_output=True)
version_text = (version_proc.stdout or "") + (version_proc.stderr or "")
print(version_text)
if version_proc.returncode != 0 or f"build {LLAMA_BUILD}" not in version_text or LLAMA_SHORT not in version_text:
    raise RuntimeError("binary version mismatch")
write_json("/kaggle/working/build_info.json", {
    "llama_cpp_commit": checked_out,
    "binary_version": version_text.strip(),
    "source_manifest": manifest,
})
'''


DOWNLOAD = r'''
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["HF_HUB_CACHE"] = "/tmp/hf/hub"
os.environ["HF_XET_CACHE"] = "/tmp/hf/xet"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download

os.makedirs(MODEL_DIR, exist_ok=True)
t0 = time.time()
downloaded = hf_hub_download(
    repo_id=HF_REPO,
    filename=HF_FILE,
    revision=HF_REVISION,
    local_dir=MODEL_DIR,
)
if os.path.abspath(downloaded) != os.path.abspath(MODEL):
    os.replace(downloaded, MODEL)
model_size = os.path.getsize(MODEL)
print("downloaded model:", MODEL, model_size, f"{model_size/2**30:.3f} GiB", f"in {time.time()-t0:.1f}s")
if model_size != MODEL_BYTES:
    raise RuntimeError(f"model size mismatch: {model_size} != {MODEL_BYTES}")
t0 = time.time()
model_sha = sha256_file(MODEL, chunk=32 << 20)
print("model sha256:", model_sha, f"in {time.time()-t0:.1f}s")
if model_sha.lower() != MODEL_SHA256:
    raise RuntimeError("model SHA-256 mismatch")
if os.path.islink(MODEL):
    materialized = MODEL + ".materialized"
    shutil.copyfile(os.path.realpath(MODEL), materialized)
    os.replace(materialized, MODEL)
shutil.rmtree("/tmp/hf", ignore_errors=True)
shutil.rmtree(MODEL_DIR + "/.cache", ignore_errors=True)
write_json("/kaggle/working/model_info.json", {
    "repo_id": HF_REPO,
    "filename": HF_FILE,
    "revision": HF_REVISION,
    "bytes": model_size,
    "sha256": model_sha,
})
print(subprocess.run("df -h /kaggle/working /tmp", shell=True, text=True, capture_output=True).stdout)
'''


PROMPT = r'''
prompt_proc = subprocess.run([DUMP, MODEL], text=True, capture_output=True, timeout=600)
prompt_text = (prompt_proc.stdout or "") + (prompt_proc.stderr or "")
open("/kaggle/working/prompt_dump.log", "w", encoding="utf-8").write(prompt_text)
print(prompt_text[-5000:])
add_match = re.search(r"add_bos=(\d+)\s+bos=(-?\d+)", prompt_text)
first_match = re.search(r"tokens add_special=true\s+\(\d+\):\s+(-?\d+)", prompt_text)
add_bos = int(add_match.group(1)) if add_match else None
bos_id = int(add_match.group(2)) if add_match else None
first_token = int(first_match.group(1)) if first_match else None
prompt_ok = (
    prompt_proc.returncode == 0
    and add_bos == 1
    and bos_id == 2
    and first_token == 2
    and "bos piece '<bos>' present in prompt text: no" in prompt_text
)
write_json("/kaggle/working/prompt_check.json", {
    "passed": prompt_ok,
    "returncode": prompt_proc.returncode,
    "add_bos": add_bos,
    "bos_id": bos_id,
    "first_token": first_token,
})
print("PROMPT CHECK:", "PASS" if prompt_ok else "FAIL")
'''


EASY = r'''
EASY_QUESTIONS = [
    ("hamlet", "Who wrote the play Hamlet?"),
    ("paris", "What is the capital of France?"),
    ("math", "What is 17 times 3?"),
    ("monalisa", "Who painted the Mona Lisa?"),
    ("mars", "Which planet is known as the Red Planet?"),
    ("gold", "What is the chemical symbol for gold?"),
    ("ballo", "Who wrote the opera A Masked Ball?"),
    ("kipling", "Which novelist, born in Bombay in 1865, won the Nobel Prize in Literature in 1907?"),
]
EXPECTED = {
    "hamlet": {"shakespeare", "william shakespeare"},
    "paris": {"paris"},
    "math": {"51"},
    "monalisa": {"leonardo", "leonardo da vinci"},
    "mars": {"mars"},
    "gold": {"au"},
    "ballo": {"verdi", "giuseppe verdi"},
    "kipling": {"kipling", "rudyard kipling"},
}
EASY_INPUT = "/kaggle/working/easy8.jsonl"
with open(EASY_INPUT, "w", encoding="utf-8", newline="\n") as f:
    for row_id, question in EASY_QUESTIONS:
        f.write(json.dumps({"id": row_id, "question": question + "\n" + INSTRUCTION}, ensure_ascii=False) + "\n")
easy_run = run_hs("easy8", EASY_INPUT, OUT_EASY, "0", 32, 1800)
easy_rows = read_jsonl(OUT_EASY + "/meta.jsonl") if os.path.exists(OUT_EASY + "/meta.jsonl") else []
easy_correct = 0
for row in easy_rows:
    answer = (row.get("answer") or "").strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    hit = (
        row.get("status") == "ok"
        and not row.get("truncated")
        and normalized in EXPECTED.get(str(row.get("id")), set())
    )
    easy_correct += int(hit)
    print("OK" if hit else "BAD", row.get("id"), repr(answer[:160]))
print(f"easy correct: {easy_correct}/{len(easy_rows)}")
'''


PILOT = r'''
pilot_run = run_hs("pilot40", PILOT_INPUT, OUT_PILOT, "all", 64, 3600)
'''


GATE = r'''
import numpy as np

gate_errors = []
gate_warnings = []
pilot_log = open(pilot_run["log"], encoding="utf-8", errors="replace").read()
done_match = re.search(
    r"done:\s*(\d+)/(\d+)\s+ok=(\d+)\s+skipped=(\d+)\s+truncated=(\d+)\s+in\s+([\d.]+)\s+s\s+\(([\d.]+)\s+ex/s\)",
    pilot_log,
)
retok_match = re.search(r"retokenize mismatch:\s*(\d+)/(\d+)", pilot_log)
pilot_cfg_path = OUT_PILOT + "/config.json"
pilot_meta_path = OUT_PILOT + "/meta.jsonl"
pilot_cfg = json.load(open(pilot_cfg_path, encoding="utf-8")) if os.path.exists(pilot_cfg_path) else {}
pilot_rows = read_jsonl(pilot_meta_path) if os.path.exists(pilot_meta_path) else []

if not prompt_ok:
    gate_errors.append("prompt check failed")
if easy_run["rc"] != 0 or len(easy_rows) != 8 or easy_correct < 7:
    gate_errors.append(f"easy gate failed: rc={easy_run['rc']} rows={len(easy_rows)} correct={easy_correct}")
easy_bad_status = sum(row.get("status") != "ok" for row in easy_rows)
easy_truncated = sum(bool(row.get("truncated")) for row in easy_rows)
easy_empty = sum(not (row.get("answer") or "").strip() for row in easy_rows)
if easy_bad_status or easy_truncated > 1 or easy_empty:
    gate_errors.append(
        f"easy structure failed: bad_status={easy_bad_status} truncated={easy_truncated} empty={easy_empty}"
    )
if not done_match:
    gate_errors.append("pilot done summary missing")
else:
    done, total, ok, skipped, truncated = map(int, done_match.groups()[:5])
    seconds, ex_per_s = float(done_match.group(6)), float(done_match.group(7))
    if (done, total, ok, skipped) != (N_PILOT, N_PILOT, N_PILOT, 0):
        gate_errors.append(f"pilot counts: {done}/{total} ok={ok} skipped={skipped}")
    if truncated > math.floor(N_PILOT * 0.10):
        gate_errors.append(f"pilot truncated {truncated}/{N_PILOT}")
    estimated_hours = N_MAIN / ex_per_s / 3600 if ex_per_s > 0 else float("inf")
    if estimated_hours >= 4.0:
        gate_errors.append(f"full estimate too high: {estimated_hours:.2f} h")
    easy_log_text = open(easy_run["log"], encoding="utf-8", errors="replace").read()
    easy_done_match = re.search(r"done:.*?in\s+([\d.]+)\s+s", easy_log_text)
    easy_compute_s = float(easy_done_match.group(1)) if easy_done_match else 0.0
    model_load_s = max(0.0, easy_run["elapsed_s"] - easy_compute_s)
    pilot_compute_s = max(0.0, pilot_run["elapsed_s"] - model_load_s)
    wall_estimated_hours = pilot_run["elapsed_s"] * (N_MAIN / N_PILOT) / 3600
    adjusted_wall_estimated_hours = (model_load_s + pilot_compute_s * (N_MAIN / N_PILOT)) / 3600
    if adjusted_wall_estimated_hours >= 4.0:
        gate_errors.append(f"adjusted wall-clock estimate too high: {adjusted_wall_estimated_hours:.2f} h")

if pilot_run["rc"] != 0 or pilot_run["timed_out"]:
    gate_errors.append(f"pilot process failed: {pilot_run}")
if len(pilot_run["peak_mem_MiB"]) < 2 or min(pilot_run["peak_mem_MiB"]) < 1000:
    gate_errors.append(f"both GPUs were not used: {pilot_run['peak_mem_MiB']}")
if retok_match:
    retok_bad, retok_total = map(int, retok_match.groups())
    if retok_total and retok_bad / retok_total > 0.05:
        gate_errors.append(f"retokenize mismatch {retok_bad}/{retok_total}")
else:
    gate_warnings.append("pilot retokenize summary missing")

if len(pilot_rows) != N_PILOT:
    gate_errors.append(f"pilot meta rows {len(pilot_rows)}")
expected_ids = [row["id"] for row in pilot_rows_all[:N_PILOT]]
if [row.get("id") for row in pilot_rows] != expected_ids:
    gate_errors.append("pilot id order mismatch")
status_counts = collections.Counter(str(row.get("status")) for row in pilot_rows)
if status_counts != {"ok": N_PILOT}:
    gate_errors.append(f"pilot statuses {status_counts}")

answers = [(row.get("answer") or "").strip() for row in pilot_rows]
empty_answers = sum(not answer for answer in answers)
CHANNEL_MARKERS = ("<|channel|>", "<|channel>", "<channel|>", "<|start|>", "<|message|>")
META_PAT = re.compile(
    r"the answer is the answer|the correct answer is the|the prompt asks|provided text is a summary|it appears that the context",
    re.I,
)
channel_leaks = sum(any(marker in answer for marker in CHANNEL_MARKERS) for answer in answers)
meta_text = sum(bool(META_PAT.search(answer)) for answer in answers)
long_answers = sum(len(answer) > 100 for answer in answers)
repetitive = 0
considered = 0
for answer in answers:
    words = re.findall(r"\w+", answer.lower())
    if len(words) >= 10:
        considered += 1
        repetitive += int(collections.Counter(words).most_common(1)[0][1] / len(words) > 0.35)
if empty_answers:
    gate_errors.append(f"empty pilot answers: {empty_answers}")
if channel_leaks or meta_text:
    gate_errors.append(f"pilot protocol/meta leak: channel={channel_leaks} meta={meta_text}")
if long_answers / max(1, len(answers)) > 0.15:
    gate_errors.append(f"pilot long answers: {long_answers}/{len(answers)}")
if considered and repetitive / considered > 0.10:
    gate_errors.append(f"pilot repetitive answers: {repetitive}/{considered}")

for key, expected in (
    ("n_layers", EXPECTED_LAYERS),
    ("n_layers_selected", EXPECTED_LAYERS),
    ("n_embd", EXPECTED_EMBD),
    ("n_examples", N_PILOT),
    ("llama_cpp_build", LLAMA_BUILD),
):
    if pilot_cfg.get(key) != expected:
        gate_errors.append(f"pilot config {key}={pilot_cfg.get(key)} expected {expected}")
if not str(pilot_cfg.get("llama_cpp_commit", "")).startswith(LLAMA_SHORT):
    gate_errors.append(f"pilot config commit={pilot_cfg.get('llama_cpp_commit')}")
if pilot_cfg.get("template") is not True or pilot_cfg.get("thinking") is not False:
    gate_errors.append(f"pilot prompt mode template={pilot_cfg.get('template')} thinking={pilot_cfg.get('thinking')}")

pilot_array_checks = {}
for point in ("a", "c"):
    path = OUT_PILOT + f"/hidden_{point}.npy"
    if not os.path.exists(path):
        gate_errors.append(f"pilot hidden_{point}.npy missing")
        continue
    try:
        arr = np.load(path, mmap_mode="r")
        expected_shape = (N_PILOT, EXPECTED_LAYERS, EXPECTED_EMBD)
        finite = bool(np.isfinite(arr).all())
        row_nonzero = bool(np.all(np.max(np.abs(arr), axis=(1, 2)) > 0))
        layer_nonzero = bool(np.all(np.max(np.abs(arr), axis=(0, 2)) > 0))
        pilot_array_checks[point] = {
            "shape": list(arr.shape), "dtype": str(arr.dtype), "finite": finite,
            "row_nonzero": row_nonzero, "layer_nonzero": layer_nonzero,
        }
        if arr.shape != expected_shape or arr.dtype != np.float16 or not finite or not row_nonzero or not layer_nonzero:
            gate_errors.append(f"pilot hidden_{point} invalid: {pilot_array_checks[point]}")
    except Exception as exc:
        pilot_array_checks[point] = {"error": repr(exc)}
        gate_errors.append(f"pilot hidden_{point} unreadable: {exc}")

bytes_per_main_array = N_MAIN * EXPECTED_LAYERS * EXPECTED_EMBD * 2
working_free_after_pilot = shutil.disk_usage("/kaggle/working").free
required_working = 2 * bytes_per_main_array + 2 * 2**30
if working_free_after_pilot < required_working:
    gate_errors.append(
        f"insufficient output disk: free={working_free_after_pilot/2**30:.2f} GiB "
        f"required={required_working/2**30:.2f} GiB"
    )

gate_passed = not gate_errors
gate_report = {
    "passed": gate_passed,
    "errors": gate_errors,
    "warnings": gate_warnings,
    "model": {"repo": HF_REPO, "file": HF_FILE, "revision": HF_REVISION, "sha256": MODEL_SHA256},
    "llama_cpp": {"commit": LLAMA_COMMIT, "build": LLAMA_BUILD},
    "prompt_ok": prompt_ok,
    "easy_correct": easy_correct,
    "pilot_run": pilot_run,
    "pilot_config": pilot_cfg,
    "pilot_arrays": pilot_array_checks,
    "pilot_answers": {
        "empty": empty_answers, "channel_leaks": channel_leaks, "meta_text": meta_text,
        "long": long_answers, "repetitive": repetitive, "considered_long": considered,
        "samples": answers[:12],
    },
    "estimated_full_hours": estimated_hours if done_match else None,
    "wall_estimated_full_hours": wall_estimated_hours if done_match else None,
    "adjusted_wall_estimated_full_hours": adjusted_wall_estimated_hours if done_match else None,
    "estimated_model_load_seconds": model_load_s if done_match else None,
    "working_free_GiB": working_free_after_pilot / 2**30,
    "main_array_GiB_each": bytes_per_main_array / 2**30,
}
write_json("/kaggle/working/gate.json", gate_report)
print(json.dumps(gate_report, ensure_ascii=False, indent=2))
print("GATE:", "PASS" if gate_passed else "FAIL")
'''


MAIN = r'''
main_run = None
if gate_passed:
    main_run = run_hs("main_" + TAG, MAIN_SOURCE, OUT_MAIN, "all", 64, 5 * 3600)
else:
    open("/kaggle/working/GATE_FAILED.txt", "w", encoding="utf-8").write(
        "Full run was not started.\n" + "\n".join(gate_errors) + "\n"
    )
    print("full run skipped by gate")
'''


FINAL = r'''
import numpy as np

summary = {
    "tag": TAG,
    "gate": gate_report,
    "main_started": main_run is not None,
    "main_run": main_run,
    "main_valid": False,
    "main_errors": [],
}
main_errors = summary["main_errors"]

if main_run is not None:
    if main_run["rc"] != 0 or main_run["timed_out"]:
        main_errors.append(f"main process failed: {main_run}")
    main_meta_path = OUT_MAIN + "/meta.jsonl"
    main_cfg_path = OUT_MAIN + "/config.json"
    main_rows = read_jsonl(main_meta_path) if os.path.exists(main_meta_path) else []
    main_cfg = json.load(open(main_cfg_path, encoding="utf-8")) if os.path.exists(main_cfg_path) else {}
    summary["main_config"] = main_cfg
    summary["main_meta_rows"] = len(main_rows)

    if len(main_rows) != N_MAIN:
        main_errors.append(f"main meta rows {len(main_rows)} != {N_MAIN}")
    if [row.get("row") for row in main_rows] != list(range(len(main_rows))):
        main_errors.append("main row indices are not contiguous")
    if [row.get("id") for row in main_rows] != [row.get("id") for row in main_rows_all]:
        main_errors.append("main id order mismatch")
    status_counts = collections.Counter(str(row.get("status")) for row in main_rows)
    if status_counts != {"ok": N_MAIN}:
        main_errors.append(f"main statuses {status_counts}")

    truncated = sum(bool(row.get("truncated")) for row in main_rows)
    answers = [(row.get("answer") or "").strip() for row in main_rows]
    empty = sum(not answer for answer in answers)
    channel_leaks = sum(any(marker in answer for marker in CHANNEL_MARKERS) for answer in answers)
    meta_text = sum(bool(META_PAT.search(answer)) for answer in answers)
    long_answers = sum(len(answer) > 100 for answer in answers)
    repetitive = 0
    considered = 0
    for answer in answers:
        words = re.findall(r"\w+", answer.lower())
        if len(words) >= 10:
            considered += 1
            repetitive += int(collections.Counter(words).most_common(1)[0][1] / len(words) > 0.35)
    summary["main_answers"] = {
        "truncated": truncated, "empty": empty, "channel_leaks": channel_leaks,
        "meta_text": meta_text, "long": long_answers, "repetitive": repetitive,
        "considered_long": considered, "samples": answers[:12],
    }
    if truncated / max(1, len(main_rows)) > 0.10:
        main_errors.append(f"main truncated {truncated}/{len(main_rows)}")
    if empty / max(1, len(main_rows)) > 0.02:
        main_errors.append(f"main empty answers {empty}/{len(main_rows)}")
    if channel_leaks or meta_text:
        main_errors.append(f"main protocol/meta leak channel={channel_leaks} meta={meta_text}")
    if long_answers / max(1, len(main_rows)) > 0.15:
        main_errors.append(f"main long answers {long_answers}/{len(main_rows)}")
    if considered and repetitive / considered > 0.10:
        main_errors.append(f"main repetitive answers {repetitive}/{considered}")

    for key, expected in (
        ("n_layers", EXPECTED_LAYERS),
        ("n_layers_selected", EXPECTED_LAYERS),
        ("n_embd", EXPECTED_EMBD),
        ("n_examples", N_MAIN),
        ("llama_cpp_build", LLAMA_BUILD),
    ):
        if main_cfg.get(key) != expected:
            main_errors.append(f"main config {key}={main_cfg.get(key)} expected {expected}")
    if not str(main_cfg.get("llama_cpp_commit", "")).startswith(LLAMA_SHORT):
        main_errors.append(f"main config commit={main_cfg.get('llama_cpp_commit')}")

    main_log = open(MAIN_LOG, encoding="utf-8", errors="replace").read() if os.path.exists(MAIN_LOG) else ""
    done_match = re.search(
        r"done:\s*(\d+)/(\d+)\s+ok=(\d+)\s+skipped=(\d+)\s+truncated=(\d+)\s+in\s+([\d.]+)\s+s\s+\(([\d.]+)\s+ex/s\)",
        main_log,
    )
    retok_match = re.search(r"retokenize mismatch:\s*(\d+)/(\d+)", main_log)
    summary["main_log_summary"] = done_match.groups() if done_match else None
    summary["main_retokenize"] = retok_match.groups() if retok_match else None
    if not done_match or tuple(map(int, done_match.groups()[:4])) != (N_MAIN, N_MAIN, N_MAIN, 0):
        main_errors.append("main done summary missing or inconsistent")
    elif int(done_match.group(5)) != truncated:
        main_errors.append(f"main log/meta truncated mismatch {done_match.group(5)} != {truncated}")
    if retok_match:
        bad, total = map(int, retok_match.groups())
        if total and bad / total > 0.05:
            main_errors.append(f"main retokenize mismatch {bad}/{total}")

    array_reports = {}
    expected_shape = (N_MAIN, EXPECTED_LAYERS, EXPECTED_EMBD)
    for point in ("a", "c"):
        path = OUT_MAIN + f"/hidden_{point}.npy"
        report = {"path": path, "exists": os.path.exists(path)}
        array_reports[point] = report
        if not os.path.exists(path):
            main_errors.append(f"main hidden_{point}.npy missing")
            continue
        try:
            arr = np.load(path, mmap_mode="r")
            report["shape"] = list(arr.shape)
            report["dtype"] = str(arr.dtype)
            report["bytes"] = os.path.getsize(path)
            if arr.shape != expected_shape or arr.dtype != np.float16:
                main_errors.append(f"main hidden_{point} shape/dtype invalid: {report}")
                continue
            finite = True
            zero_rows = []
            layer_max = np.zeros(EXPECTED_LAYERS, dtype=np.float64)
            for start in range(0, N_MAIN, 64):
                block = np.asarray(arr[start:start + 64], dtype=np.float32)
                finite = finite and bool(np.isfinite(block).all())
                row_max = np.max(np.abs(block), axis=(1, 2))
                zero_rows.extend((start + int(i)) for i in np.flatnonzero(row_max == 0))
                layer_max = np.maximum(layer_max, np.max(np.abs(block), axis=(0, 2)))
            report["finite"] = finite
            report["zero_rows"] = zero_rows[:20]
            report["zero_layers"] = [int(i) for i in np.flatnonzero(layer_max == 0)]
            if not finite or zero_rows or report["zero_layers"]:
                main_errors.append(f"main hidden_{point} numeric validation failed: {report}")
        except Exception as exc:
            report["error"] = repr(exc)
            main_errors.append(f"main hidden_{point} unreadable: {exc}")
    summary["main_arrays"] = array_reports
    summary["main_valid"] = not main_errors

write_json("/kaggle/working/run_summary.json", summary)
if summary["main_valid"]:
    open("/kaggle/working/SUCCESS.txt", "w", encoding="utf-8").write(
        f"{TAG} completed and passed structural validation.\n"
    )
else:
    open("/kaggle/working/FAILED.txt", "w", encoding="utf-8").write(
        "\n".join(main_errors or gate_errors or ["main was not started"]) + "\n"
    )
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(subprocess.run("du -sh /kaggle/working/out /kaggle/working/* 2>/dev/null | tail -30", shell=True, text=True, capture_output=True).stdout)
'''


manifest_path = Path(__file__).resolve().parent / "_build" / "ds" / "source-manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing {manifest_path}; run push.ps1 -DryRun or push.ps1 first")
with open(manifest_path, encoding="utf-8") as f:
    expected_source_manifest = json.load(f)
SETUP = "EXPECTED_SOURCE_MANIFEST = " + repr(expected_source_manifest) + "\n" + SETUP

cells = [
    code("setup", SETUP),
    code("build", BUILD),
    code("download", DOWNLOAD),
    code("prompt", PROMPT),
    code("easy", EASY),
    code("pilot", PILOT),
    code("gate", GATE),
    code("main", MAIN),
    code("final", FINAL),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_dir = Path(__file__).resolve().parent / "_build" / "kernel"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "hs26b.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

metadata = {
    "id": KERNEL_ID,
    "title": "hs gemma4 26b a4b",
    "code_file": "hs26b.ipynb",
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
with open(out_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

for cell in cells:
    compile("".join(cell["source"]), f"<cell:{cell['id']}>", "exec")
print("wrote and syntax-checked", len(cells), "cells to", out_dir)
