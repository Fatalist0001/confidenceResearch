# Invalid LFM3 diagnostic input

This is not a 20-question result. `pipeline/data/diag_lfm3_sampler20.jsonl` used
literal backslash-n separators instead of physical JSONL newlines. The extractor
received only one example; `config.json` and `meta.jsonl` confirm N=1.

The configured answer budget was 256 and sampling was `common_sampler`, but the
single untruncated answer says nothing about completion on the intended 20 rows.
Do not quote it as 0/20 truncated.

The corrected input is `pipeline/data/diag_lfm3_sampler20_v2.jsonl`; its separate
output directory is `diag_lfm3_sampler20_n256_v2`. Keep this original input and
output unchanged as evidence.
