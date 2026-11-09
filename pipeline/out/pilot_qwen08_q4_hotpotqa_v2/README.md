# Qwen3.5-0.8B Q4_K_M: HotpotQA technical pilot

2026-11-09. Profile `qwen-hotpotqa`, tag `qwen08_q4_hotpotqa_v2`.

- 200 disjoint technical-pilot questions; evaluation uses the next 500 hash-ranked
  IDs, with no overlap or replacement.
- Every supplied paragraph stays in dataset order, including distractors. Two
  source records have fewer than the nominal ten paragraphs; the original JSON
  audit confirmed that this is not a conversion loss.
- Stock non-thinking template, raw argmax, 64 answer tokens, max_len 8192,
  batch_rows 1, model layer 13 only. Binary `785322d17`, build 12033.
- 200/200 generated and captured; no skipped, truncated or empty answers, protocol
  leaks, non-finite or zero state rows. Both arrays are float16 `[200,1,1024]`.
- Official whole-answer EM: 59/200 = 0.295. Mean answer F1: 0.4097 (rounded).
  Labels are `hotpot_em`/`hotpot_f1`, not TriviaQA substring labels.
- Prelabel and final class-balance gates passed, 0 failures and 0 warnings.
  Full state verification passed, 0 failures and 0 warnings.
- Binary time 90.7 s, 2.20 questions/s; predicted 500-question time about 0.063 h.
  Runner elapsed time also includes provenance checks, labeling and verification.

No probe is trained or scored on this pilot. The evaluation applies the existing
`main_qwen08_q4/probe_model.npz` C/model-layer-13 probe with its original train
standardization and lambda 1000. The pilot is not a research test result.

Protocol: `pipeline/data/hotpotqa_qwen08_v2/protocol.json`, SHA-256
`f0f1f88cfc277247d97c37f2acf8ec8c0728a7b85a9121e8c9fcbf5d830d0657`.
See `docs/HOTPOTQA-TRANSFER.md` and `docs/HOTPOTQA-SOURCE-AUDIT.md` for provenance,
exact sampling, scoring rules and the source-versus-target runtime caveat.

All outputs are final; do not resume or overwrite this directory.
