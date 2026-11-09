# LFM3 corrected 20-question sampler diagnostic

Historical diagnostic evidence, not part of the HotpotQA transfer test.

- Input: `pipeline/data/diag_lfm3_sampler20_v2.jsonl`, 20 actual JSONL records.
- Model: LFM3-1.7B QAD Q4, stock non-thinking template.
- `common_sampler`: temperature 0.1, top_k 50, repeat_penalty 1.1, seed 42+row.
- Answer budget 256; metadata contains 20 rows, 9 truncated answers.

This diagnostic did not resolve the completion problem. It is not the proposed
paired stock `llama-cli` control, and does not authorize a new full LFM3 run.
Do not modify the original output or confuse it with the invalid N=1 directory
`diag_lfm3_sampler20_n256`.
