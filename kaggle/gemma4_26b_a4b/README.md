# Gemma 4 26B-A4B on Kaggle

Private kernel: `odinodinodin/hs-gemma4-26b-a4b`.

Model:

- repository: `unsloth/gemma-4-26B-A4B-it-GGUF`;
- file: `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`;
- HF revision: `c099eb48e663fd284577b04978a94ffccb261841`;
- size: 16,947,541,728 bytes (15.78 GiB);
- SHA-256: `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`.

The notebook builds `llama-hidden-states` at llama.cpp
`785322d17f2d51e5185cf996a1f4e46d87640841` (build 12033) and runs on two T4 GPUs
with `-sm layer -ts 1,1 -ngl 99`.

Autonomous sequence:

1. verify both GPUs, free disk, dataset hashes and the 200/6000 inputs;
2. build the pinned binary and `hs-prompt-dump`;
3. download and hash the pinned GGUF;
4. check BOS (`add_bos=1`, BOS and first token are 2);
5. easy8;
6. pilot40 with all layers;
7. gate answers, speed, both-GPU usage, shapes, finite/non-zero states and output disk;
8. main6000 only after the gate;
9. full structural and numeric validation, `run_summary.json`, `SUCCESS.txt` or
   `FAILED.txt`.

Launch without a background watcher:

```powershell
powershell -File kaggle\gemma4_26b_a4b\push.ps1
```

Later download into a new version directory:

```powershell
$KAG = 'C:\Windows\Temp\opencode\kgvenv\Scripts\kaggle.exe'
& $KAG kernels output odinodinodin/hs-gemma4-26b-a4b `
  -p kaggle\gemma4_26b_a4b\out\v<N>
```

## Versions

- v1 is invalid: build, model hash, BOS and easy8 completed, then the notebook stopped
  before pilot on a local variable typo (`EXPECT` instead of `EXPECTED`). No pilot or
  main artifacts were produced. Evidence is kept under `out/failure-v1*`.
- v2 fixes the typo and adds exact-match easy-answer scoring plus an adjusted wall-time
  estimate that removes the measured one-time model-load cost.

## Result v2 (2026-11-04)

- easy8: 7/8; pilot40 passed the gate in 21.0 s;
- main: 6000/6000, 32 truncated, 6 retokenize mismatch;
- generation/state pass: 1866.7 s, 3.21 examples/s;
- both T4 GPUs used (peak 8859/9355 MiB, peak utilization 75/74%);
- arrays: `[6000,30,2816]` float16, 1,013,760,128 bytes (0.944 GiB) each;
- finite states, no zero rows or layers; Kaggle structural validation passed;
- `filtered_b=0.708`; baseline test AUROC 0.838 (logprob), 0.843 (entropy);
- frozen probe: λ=3000, C/16, val AUROC 0.9018, test AUROC **0.9077**
  [0.8893, 0.9235], test AUPRC 0.9576; `probe2` reproduced both CSVs byte-for-byte.

The output was downloaded on 2026-11-05 into the new directory
`out/v2-retrieved-20261105` and copied, without modifying the Kaggle copy, to
`pipeline/out/main_gemma4_26b_a4b_udq4km` for local label/verify/baseline/probe stages.
The two large files downloaded at about 8 MiB/s; total output size is 1.906 GiB.
