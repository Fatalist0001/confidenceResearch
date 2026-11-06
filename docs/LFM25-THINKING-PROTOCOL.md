# LFM2.5-1.2B-Thinking: frozen reasoning protocol v1

Дата фиксации: 2026-11-05. Этот режим отделён от обычного frozen probe protocol:
генерация состоит из reasoning, final и отдельного replay состояний. λ-сетка, splits,
метка `correct_filtered_b`, выбор по val и однократный test не меняются.

## Выбор reasoning budget

Budget выбирался только на первых 200 строках `train` из `pipeline/data/main.jsonl`.
Val и test при выборе не использовались. Кандидаты и заранее заданное правило:

- reasoning budget: 32, 64, 128, 256 токенов;
- final budget: 64 токена для всех кандидатов;
- coverage не ниже 95%;
- выбрать минимальный budget, чья effective accuracy (положительные
  `correct_filtered_b` / все 200 строк) отстаёт от лучшего кандидата не более чем на
  0.02.

| reasoning budget | final coverage | `correct_filtered_b` / 200 | время reasoning / 200 |
|---:|---:|---:|---:|
| 32 | 199/200 = 0.995 | 57/200 = 0.285 | 47.9 s |
| 64 | 199/200 = 0.995 | 59/200 = 0.295 | 86.7 s |
| 128 | 200/200 = 1.000 | 59/200 = 0.295 | 167.6 s |
| 256 | 200/200 = 1.000 | 60/200 = 0.300 | 325.7 s |

Выбран **reasoning budget 32**: он проходит coverage-гейт и уступает максимуму 0.015,
но reasoning-проход примерно в 6.8 раза быстрее 256. Машиночитаемый результат —
`pipeline/out/lfm25_thinking_budget_selection.json`.

## Генерация

1. Stock chat template генерирует не более 32 reasoning-токенов raw greedy.
2. Если модель сама выдала `</think>`, всё после первого маркера отбрасывается.
3. Иначе reasoning принудительно закрывается. К контексту добавляется ровно
   `</think>\nFinal answer:`.
4. Из этого контекста raw greedy генерирует не более 64 final-токенов.
5. Final заканчивается на первом из событий:
   - новая строка;
   - естественный EOG;
   - специальный маркер `<think`, `</think` или `<|`;
   - возобновление рассуждения по regex
     `\s+(wait|however|alternatively|but|hmm)\b` без учёта регистра.
6. Пустой final остаётся пустым и получает null-метку; reasoning никогда не участвует
   в разметке ответа.

`scripts/prepare_lfm_thinking.ps1` создаёт reasoning и final evidence, проверяет coverage
и пишет подготовленный replay-input. Существующие каталоги и файлы не перезаписываются.

## Возобновление после остановки

Если reasoning-проход прерван, повторный запуск делается с `-Resume`:

```powershell
powershell -File scripts\prepare_lfm_thinking.ps1 -Stage main -Resume -Model <gguf>
```

Скрипт читает только завершённые строки `meta.jsonl` (обрезанный хвост последней
записи игнорируется), строит вход из оставшихся глобальных строк, генерирует их в новый
каталог `..._reasoning_rest_from_<N>` и сливает обе части в
`..._protocol/reasoning_meta.jsonl` с восстановленными индексами `row`. Исходный
частичный каталог не изменяется.

## Точки состояний и baseline

Replay выполняется без генерации по полному контексту
`user prompt + reasoning + </think> + Final answer:` и извлечённому final:

- **A** — последний токен `Final answer:`, состояние после reasoning, из которого
  выдаётся первый токен final;
- **C** — последний содержательный токен извлечённого final.

Logprob и entropy первого final-токена берутся из отдельного final-generation pass.
Раннер сохраняет исходный replay-meta как `meta.raw.jsonl`, затем создаёт итоговый
`meta.jsonl` с этой provenance. Поэтому A baseline и A hidden state относятся к одной
позиции, а C не попадает в последующее повторное рассуждение.

При сравнении с LFM2.5-1.2B-Instruct точка C сопоставима напрямую. Точка A имеет другой
смысл: у Thinking она измерена **после** фиксированного reasoning, у Instruct — сразу
после пользовательского prompt. Это различие должно сопровождать любую таблицу или
вывод по A.

## Результат main (2026-11-06)

| | значение |
|---|---|
| main replay | 6000/6000, 0 skipped, 0 truncated, 0 retokenize mismatch |
| пустые final | 39/6000 = 0.0065 |
| replay | 128.3 s, 50.18 ex/s |
| подготовка reasoning/final | ≈1.11 ч |
| `filtered_b` | 1157/5961 = 0.194 |
| baseline test logprob / entropy | 0.634 / 0.662 |
| λ / точка / слой | 1000 / C / 9 |
| val AUROC / AUPRC | 0.8819 / 0.6443 |
| test AUROC | **0.8813** [0.8553, 0.9051] |
| test AUPRC | 0.6491 |
| Brier / ECE | 0.0940 / 0.0328 |
| shuffled / layer 0 C test | 0.480 ± 0.047 / 0.716 |
| `probe2` | `probe_curve.csv` и `probe_scores.csv` совпали побайтово |
| verify / gate | 0 failed, 0 warnings / passed, 1 ожидаемое предупреждение |

Final-извлечение: новая строка — 1814, возобновление рассуждения — 4147, нет stop-сигнала
— 39; естественных закрытий reasoning за 32 токена — 0.

Сравнение с Instruct на 1190 общих test-строках (base rate 0.179 / 0.176, метки
различаются на 183 строках): ΔAUROC **+0.0016 [-0.0278, +0.0316]**, ΔAUPRC
**−0.0365 [−0.1019, +0.0282]**. На val: ΔAUROC −0.0188 [−0.0471, +0.0084], ΔAUPRC
−0.0587 [−0.1331, +0.0099]. Оба test-интервала накрывают ноль.

Артефакты: `pipeline/out/main_lfm25_1p2b_thinking_r32_q4_v2`; сравнения
`pipeline/out/compare_main_lfm25_1p2b_instruct_q4_lfm25_1p2b_thinking_r32_q4_v2_{test,val}.json`;
выбор budget — `pipeline/out/lfm25_thinking_budget_selection.json`; входы
`pipeline/data/{pilot,main}_lfm25_1p2b_thinking_r32_q4_v2_final.jsonl`.
