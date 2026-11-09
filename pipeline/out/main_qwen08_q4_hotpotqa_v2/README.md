# Qwen3.5-0.8B Q4_K_M: frozen-probe HotpotQA transfer

**Outcome: no advantage over first-token confidence was demonstrated.** The frozen
probe retains above-chance ranking, but its point estimates are below both
baselines. Paired intervals include zero, so this sample also does not establish
that the probe is worse.

2026-11-09. Profile `qwen-hotpotqa`; protocol and source audit are described in
`docs/HOTPOTQA-TRANSFER.md` and `docs/HOTPOTQA-SOURCE-AUDIT.md`.

## Protocol and integrity

- 500 evaluation questions, disjoint from the 200-question pilot; ID-only SHA-256
  selection with seed 42, no replacements.
- All supplied paragraphs and distractors in original order, unchanged sentence
  strings. Seven evaluation records have fewer than the nominal ten paragraphs;
  the verified official JSON contains the same contexts, and the user approved
  retaining them without padding or gold-document selection.
- Stock non-thinking template, raw argmax, 64 answer tokens, max_len 8192,
  batch_rows 1, `-ngl 99`. Binary `785322d17`, build 12033.
- Source `main_qwen08_q4/probe_model.npz`: C/model layer 13, lambda 1000, original
  `w/mu/sd`. Source SHA-256 remains
  `5e948cd2146d95fef7ad2b66b21c9c5d12fdd3cfc02c250f30fcb81a1ca7088c`.
- Target array index is 0 because only model layer 13 is saved. Both hidden arrays
  are float16 `[500,1,1024]`; all values are finite. No skipped or empty answers,
  one truncated answer. Gate and full verification passed without warnings.
- No training, calibration, layer selection, search, agent tools or additional
  model-answer verification pass. There are no target `probe.*` or `baseline.*`
  fitting artifacts.

## Official answer accuracy

All 500 attempts are evaluated as whole raw answers, including the truncated one;
there is no favorable answer extraction or substring matching.

| metric | value | bootstrap 95% CI |
|---|---:|---:|
| answer EM | 161/500 = 0.3220 | [0.2840, 0.3660] |
| mean answer F1 | 0.4286 | [0.3902, 0.4682] |
| completed/primary coverage | 499/500 = 0.998 | descriptive |

The common primary cohort has 499 completed non-empty answers, 161 positives,
valid C states and first-token scores. Row 247 (`5add23d05542994734353871`) is
truncated at 64 tokens; it remains in all-attempt accuracy and has blank confidence
fields in `transfer_scores.csv`.

## Frozen confidence comparison

Higher score means more likely correct. Bootstrap uses 1000 question resamples,
seed 42; AUPRC is tie-grouped average precision.

| method | AUROC [95% CI] | AUPRC [95% CI] | correct among top 250 | retained EM |
|---|---:|---:|---:|---:|
| frozen C/13 probe | 0.6441 [0.5939,0.6975] | 0.4451 [0.3775,0.5305] | 106/250 | 0.424 |
| first-token logprob | 0.6655 [0.6135,0.7115] | 0.4745 [0.4056,0.5571] | 109/250 | 0.436 |
| negative first-token entropy | 0.6879 [0.6351,0.7348] | 0.4889 [0.4171,0.5753] | 112/250 | 0.448 |

Top 250 is the predeclared retained half rounded up from 499 rows (50.1% coverage),
with ties resolved by original row order. Retained accuracy is descriptive, not a
tuned threshold or a claim about verification outcomes.

| paired difference, probe minus baseline | delta AUROC [95% CI] | delta AUPRC [95% CI] |
|---|---:|---:|
| vs logprob | -0.0214 [-0.0891,+0.0525] | -0.0294 [-0.1152,+0.0627] |
| vs negative entropy | -0.0438 [-0.1136,+0.0307] | -0.0438 [-0.1341,+0.0507] |

This does not support using this frozen Qwen 0.8B probe as an improvement over
standard confidence on HotpotQA. It is one model/configuration and a preliminary
test; no conclusion about all research models or GAIA agents follows from it.

## Provenance and limitations

The source probe was produced on `3ad2230`, build 37; the target uses `785322d17`,
build 12033. Prior master control showed 75.0% answer and 97.3% label agreement.
Dataset/context, evaluation label and runtime changed, so this is not a bit-exact
dataset-only transfer. Scores on this target are nevertheless compared on exactly
the same rows.

Binary time: 348.1 s, 1.44 questions/s. Runner generation-stage elapsed time: 357.6 s;
labeling, state verification and frozen scoring are separate stages.

- `protocol.json` SHA-256: `f0f1f88cfc277247d97c37f2acf8ec8c0728a7b85a9121e8c9fcbf5d830d0657`.
- `transfer.json` SHA-256: `6b5bb46649d083d5aeb1c2f56b8a0c6b27cbd1e311646ab189ac821a9cda27cd`.
- `transfer_scores.csv` SHA-256: `57a00f562c5f84ab1f38a354a56f828f7dabed66998902220b90abc1498b17bd`.

`meta.jsonl`, `labels.csv`, both NPY arrays, gate, verify, runner and transfer outputs
are final. Do not overwrite or resume this directory. Other models and GAIA remain
deferred; no commit or push is implied by this experiment.
