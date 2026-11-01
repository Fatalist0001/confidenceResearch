# Debug-ядро: адекватность Gemma 4 12B на актуальном llama.cpp

Ядро `odinodinodin/hs-gemma4-12b-debug`, 2026-11-01, акселератор `NvidiaTeslaT4`
(2 × T4 15 ГБ), 29 мин, **0.50 ч квоты**. Собирает llama.cpp **на master** (`dde879b`,
2026-11-01) с `llama-completion`, качает `unsloth/gemma-4-12b-it-GGUF` Q4_K_M (6.63 ГиБ)
и задаёт 8 простых вопросов в нескольких конфигурациях. Наша сборка `hs` здесь не
участвует — проверяется только модель.

## Результат

| конфигурация | что подаётся | итог |
|---|---|---|
| `hand_noch` | ручной Gemma-4 промпт без пустого thought-канала | **8/8** |
| `hand_ch` | ручной промпт **с** пустым thought-каналом (как в GGUF-шаблоне) | **8/8** |
| `raw_1gpu` | без чат-шаблона, `-no-cnv`, `вопрос + "\nAnswer:"` | 3/8 |
| `tmpl_off_1gpu`, `tmpl_off_2gpu`, `tmpl_on_1gpu` | шаблон из GGUF через `--chat-template-kwargs` | **не выполнились** |

Ответы `hand_ch` (то есть ровно тот промпт, что рендерит наш инструмент):

```
William Shakespeare / Paris / 51 / Leonardo da Vinci / Mars / Au / Giuseppe Verdi / Rudyard Kipling
```

Вывод: **модель и квант исправны**; на актуальном llama.cpp 12B отвечает по существу,
и пустой thought-канал в промпте ей не мешает (оба варианта дают 8/8).

## Почему `tmpl_*` не выполнились (ошибка теста, не модели)

```
error: invalid argument: --chat-template-kwargs
```

На master `--chat-template-kwargs` зарегистрирован только для `server` и `cli`
(`common/arg.cpp`, `.set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI})`), а
`llama-completion` этот флаг отвергает. Управление «думанием» там — через
`--reasoning on|off`. Поэтому конфигурации с шаблоном из GGUF не запустились вовсе,
и нули в их строке — артефакт теста.

## Что это дало для диагноза прогона 12B

Из лога неудачного прогона (`kaggle/gemma4_12b/out/main_gemma4_12b.log`) видно, что
инструмент рендерил **канонический** промпт:

```
<|turn>user\n<вопрос> Answer with just the answer, no explanation.<turn|>\n
<|turn>model\n<|channel>thought\n<channel|>
```

Это байт-в-байт то, что в тесте соответствует `hand_ch` → 8/8. Значит **промпт и
шаблон ни при чём**; расхождение — в инференсе (версия llama.cpp и/или разбиение
модели по двум GPU). Проверка этой пары — ядро `kaggle/gemma4_12b_pin_gpu/`.

## Файлы

- `make_nb.py` — генератор ноутбука (5 ячеек: setup, build, download, tests, prompt).
- `push.ps1` — пуш ядра без датасета.
- `out/` — статус и полный лог (в git не входит).