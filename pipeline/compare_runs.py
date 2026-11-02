"""Сверка двух прогонов `llama-hidden-states` между собой (например, старая и новая
сборка llama.cpp): состояния, ответы, logprobs, метки.

Сравниваются только строки, которые есть в обоих прогонах со `status: ok`; id берутся из
meta.jsonl, строки массивов — по полю `row` (как в probe.py).

    python compare_runs.py --a out/equiv/pin_qwen08_p20 --b out/equiv/master_qwen08_p20 \
        --out out/equiv/pin_vs_master_qwen08_p20.json

Печатает:
    точки A и C: max |d|, медиана |d| (по строкам), cosine медиана/минимум;
    ответы: сколько строк совпало текстово, сколько меток `correct` (если есть labels.csv);
    logprobs: max |d| на общих позициях;
    с `--per-layer` — то же по каждому слою (видно, растёт ли дрейф с глубиной).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


def read_meta(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{i}: bad json ({e})")
    return rows


def read_labels(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_occurrences(rows):
    """Индекс по (id, номер вхождения), т.к. id в исходном JSONL не обязан быть уникальным."""
    counts = {}
    out = {}
    for row in rows:
        row_id = row["id"]
        occurrence = counts.get(row_id, 0)
        counts[row_id] = occurrence + 1
        out[(row_id, occurrence)] = row
    return out


def key_text(key):
    row_id, occurrence = key
    return row_id if occurrence == 0 else f"{row_id}#{occurrence + 1}"


def cosine_rows(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    den = na * nb
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (a * b).sum(axis=-1) / den
    cos[den == 0] = 1.0
    return cos


def compare_states(path_a, path_b, rows_a, rows_b, keys):
    """rows_* — индексы массивов для общих (id, occurrence) в порядке keys."""
    out = {}
    for name, pa, pb in (("a", os.path.join(path_a, "hidden_a.npy"), os.path.join(path_b, "hidden_a.npy")),
                         ("c", os.path.join(path_a, "hidden_c.npy"), os.path.join(path_b, "hidden_c.npy"))):
        if not (os.path.exists(pa) and os.path.exists(pb)):
            out[name] = {"missing": True}
            continue
        A = np.load(pa, mmap_mode="r")
        B = np.load(pb, mmap_mode="r")
        if A.shape[1:] != B.shape[1:]:
            out[name] = {"shape_a": list(A.shape), "shape_b": list(B.shape), "incomparable": True}
            continue
        ia = np.asarray(rows_a)
        ib = np.asarray(rows_b)
        maxd, medd, cos_values, norm_values = [], [], [], []
        for s in range(0, len(ia), 64):
            a = np.asarray(A[ia[s:s + 64]], dtype=np.float32)
            b = np.asarray(B[ib[s:s + 64]], dtype=np.float32)
            d = np.abs(a - b)
            maxd.append(d.max())
            medd.extend(np.median(d.reshape(d.shape[0], -1), axis=1))
            cos_values.extend(cosine_rows(a, b).ravel())
            norm_values.extend(np.linalg.norm(a, axis=-1).ravel())
        # медиана |d| по строкам; норму берём по A, чтобы числа были сопоставимы с отчётами
        norm_med = float(np.median(norm_values)) if norm_values else 0.0
        out[name] = {
            "shape": list(A.shape[1:]),
            "rows": len(ia),
            "max_abs": float(max(maxd)) if maxd else 0.0,
            "median_abs_per_row": float(np.median(medd)) if medd else 0.0,
            "median_abs_rel": float(np.median(medd) / norm_med) if norm_med else None,
            "state_norm_median": norm_med,
            "cosine_median": float(np.median(cos_values)) if cos_values else 0.0,
            "cosine_min": float(min(cos_values)) if cos_values else 0.0,
        }
    return out


def per_layer(path_a, path_b, rows_a, rows_b, name):
    """Пер-слойная таблица расхождения (median |d|, max |d|, cosine median)."""
    pa = os.path.join(path_a, "hidden_%s.npy" % name)
    pb = os.path.join(path_b, "hidden_%s.npy" % name)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return []
    A = np.load(pa, mmap_mode="r")
    B = np.load(pb, mmap_mode="r")
    if A.shape[1:] != B.shape[1:]:
        return []
    ia = np.asarray(rows_a)
    ib = np.asarray(rows_b)
    out = []
    for L in range(A.shape[1]):
        a = np.asarray(A[ia, L, :], dtype=np.float32)
        b = np.asarray(B[ib, L, :], dtype=np.float32)
        d = np.abs(a - b)
        norm = float(np.median(np.linalg.norm(a, axis=-1)))
        out.append({
            "layer": L,
            "median_abs": float(np.median(d)),
            "max_abs": float(d.max()),
            "rel": float(np.median(d) / norm) if norm else None,
            "cosine_median": float(np.median(cosine_rows(a, b))),
        })
    return out


def compare_meta(m_a, m_b, keys):
    same_answer = 0
    same_tokens = 0
    same_pos_a = 0
    same_pos_c = 0
    same_pos = 0
    same_trunc = 0
    lp_max = 0.0
    lp_rows = 0
    examples = []
    for key in keys:
        ra, rb = m_a[key], m_b[key]
        ans_a = (ra.get("answer") or "").strip()
        ans_b = (rb.get("answer") or "").strip()
        if ans_a == ans_b:
            same_answer += 1
        elif len(examples) < 8:
            examples.append({"id": key_text(key), "a": ans_a[:70], "b": ans_b[:70]})
        if ra.get("n_tokens") == rb.get("n_tokens"):
            same_tokens += 1
        if ra.get("pos_a") == rb.get("pos_a"):
            same_pos_a += 1
        if ra.get("pos_c") == rb.get("pos_c"):
            same_pos_c += 1
        if (ra.get("pos_a"), ra.get("pos_c")) == (rb.get("pos_a"), rb.get("pos_c")):
            same_pos += 1
        if bool(ra.get("truncated")) == bool(rb.get("truncated")):
            same_trunc += 1
        la, lb = ra.get("logprobs") or [], rb.get("logprobs") or []
        if la and lb:
            n = min(len(la), len(lb))
            lp_max = max(lp_max, float(np.max(np.abs(np.asarray(la[:n]) - np.asarray(lb[:n])))))
            lp_rows += 1
    n = len(keys)
    return {
        "rows": n,
        "same_answer": same_answer,
        "same_answer_frac": same_answer / n if n else None,
        "same_n_tokens": same_tokens,
        "same_pos_a": same_pos_a,
        "same_pos_c": same_pos_c,
        "same_pos": same_pos,
        "same_truncated": same_trunc,
        "logprob_max_abs": lp_max if lp_rows else None,
        "logprob_rows": lp_rows,
        "answer_diff_examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="каталог прогона A (эталон)")
    ap.add_argument("--b", required=True, help="каталог прогона B")
    ap.add_argument("--out", help="куда записать JSON (по умолчанию только печать)")
    ap.add_argument("--per-layer", action="store_true",
                    help="печать пер-слойной таблицы расхождения (точка A)")
    args = ap.parse_args()

    ra = read_meta(os.path.join(args.a, "meta.jsonl"))
    rb = read_meta(os.path.join(args.b, "meta.jsonl"))
    ma = index_occurrences(ra)
    mb = index_occurrences(rb)

    keys = [key for key, row in ma.items() if row.get("status") == "ok"
            and key in mb and mb[key].get("status") == "ok"]
    only_a = [key_text(key) for key in ma if key not in mb]
    only_b = [key_text(key) for key in mb if key not in ma]
    if not keys:
        sys.exit("нет общих ok-строк — сравнивать нечего")

    rows_a = [int(ma[key]["row"]) for key in keys]
    rows_b = [int(mb[key]["row"]) for key in keys]

    res = {
        "a": args.a,
        "b": args.b,
        "common_ok_rows": len(keys),
        "only_in_a": only_a[:10],
        "only_in_b": only_b[:10],
        "n_only_in_a": len(only_a),
        "n_only_in_b": len(only_b),
        "states": compare_states(args.a, args.b, rows_a, rows_b, keys),
        "meta": compare_meta(ma, mb, keys),
    }
    if args.per_layer:
        res["per_layer_a"] = per_layer(args.a, args.b, rows_a, rows_b, "a")

    labels_a = read_labels(os.path.join(args.a, "labels.csv"))
    labels_b = read_labels(os.path.join(args.b, "labels.csv"))
    if labels_a and labels_b:
        la = index_occurrences(labels_a)
        lb = index_occurrences(labels_b)
        label_keys = [key for key in keys if key in la and key in lb]
        label_column = "correct_filtered_b" if "correct_filtered_b" in labels_a[0] else "correct_strict"
        same = sum(1 for key in label_keys if la[key].get(label_column) == lb[key].get(label_column))
        res["labels"] = {"key": label_column, "same": same, "rows": len(label_keys),
                         "same_frac": same / len(label_keys) if label_keys else None}
        if "correct_strict" in labels_a[0]:
            k2 = "correct_strict"
            same2 = sum(1 for key in label_keys if la[key].get(k2) == lb[key].get(k2))
            res["labels"]["same_strict"] = same2
            res["labels"]["same_strict_frac"] = same2 / len(label_keys) if label_keys else None

    print(f"общих ok-строк: {len(keys)}  (только в A: {len(only_a)}, только в B: {len(only_b)})")
    for name, label in (("a", "точка A"), ("c", "точка C")):
        s = res["states"].get(name, {})
        if s.get("missing"):
            print(f"{label}: массив отсутствует")
        elif s.get("incomparable"):
            print(f"{label}: формы не совпадают {s['shape_a']} vs {s['shape_b']}")
        else:
            print(f"{label}: max|d|={s['max_abs']:.3e}  медиана |d|={s['median_abs_per_row']:.3e} "
                  f"({100 * (s['median_abs_rel'] or 0):.2f} % от нормы {s['state_norm_median']:.2f})  "
                  f"cosine медиана={s['cosine_median']:.6f} мин={s['cosine_min']:.6f}")
    m = res["meta"]
    print(f"ответы: текстово совпали {m['same_answer']}/{m['rows']} "
          f"({100 * (m['same_answer_frac'] or 0):.1f} %), n_tokens {m['same_n_tokens']}/{m['rows']}, "
          f"pos_a {m['same_pos_a']}/{m['rows']}, pos_c {m['same_pos_c']}/{m['rows']}, "
          f"truncated {m['same_truncated']}/{m['rows']}")
    if m["logprob_max_abs"] is not None:
        print(f"logprobs: max|d| = {m['logprob_max_abs']:.3e} на {m['logprob_rows']} строках")
    if args.per_layer and res.get("per_layer_a"):
        print("пер-слойно (точка A):  слой  медиана|d|    max|d|    % нормы  cosine")
        for L in res["per_layer_a"]:
            print(f"                        {L['layer']:4d}  {L['median_abs']:.3e}  "
                  f"{L['max_abs']:.3e}  {100 * (L['rel'] or 0):6.3f}  {L['cosine_median']:.6f}")
    if "labels" in res:
        lab = res["labels"]
        line = f"метки ({lab['key']}): {lab['same']}/{lab['rows']} совпали"
        if "same_strict" in lab:
            line += f"; strict {lab['same_strict']}/{lab['rows']}"
        print(line)
    if m["answer_diff_examples"]:
        print("примеры разных ответов:")
        for e in m["answer_diff_examples"]:
            print(f"   {e['id']:12s} A={e['a']!r}  B={e['b']!r}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
