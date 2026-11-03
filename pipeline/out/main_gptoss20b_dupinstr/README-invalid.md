# Invalid run: duplicated instruction in the prompt

Status: **invalid. Do not cite any number from this directory.**

This directory was produced on 2026-10-22 with the first gpt-oss-20b input. The
instruction `Answer with just the answer, no explanation.` was already part of
every `question` in `data/main.jsonl`, and `pipeline/make_gptoss_prompt.py` added
it a second time inside the harmony prompt. The model saw it twice.

Why this matters more than the numbers look: the probe, the baseline and the
labels in this directory are complete and plausible (they were computed before
anyone noticed the duplicate), so anything that reads `probe.json` from here will
produce a real-looking result for a prompt that was never intended.

The valid replacement is `pipeline/out/main_gptoss20b_v2`
(`probe_status: complete` in `RUNS.json`).

Files are kept as evidence of the incident. They are not regenerated and not
deleted: see `docs/history/AGENTS-full-2026-11-02.md`, section
"Инцидент 2026-10-22".
