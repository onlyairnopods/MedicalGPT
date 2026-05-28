#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select a LARGE subset (e.g. 100K) from FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K
for continued SFT / repair-SFT.

Why this v2 exists:
- v1 was tuned for smaller, cleaner subsets (16K / 24K / 30K).
- For 100K, we need looser filters and a guaranteed global fill stage.

Usage:
python huatuo_select_for_repair_sft_v2.py \
  --output selected_huatuo_100k.jsonl \
  --target-size 100000

Balanced mode:
python huatuo_select_for_repair_sft_v2.py \
  --output selected_huatuo_100k.jsonl \
  --target-size 100000 \
  --selection-mode balanced

Global score mode:
python huatuo_select_for_repair_sft_v2.py \
  --output selected_huatuo_100k.jsonl \
  --target-size 100000 \
  --selection-mode global
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "assistant"}

# DEFAULT_BUCKET_WEIGHTS = {
#     "treatment": 0.24,
#     "diagnosis": 0.18,
#     "medical_knowledge": 0.18,
#     "drug": 0.10,
#     "exam_style": 0.08,
#     "check_report": 0.06,
#     "triage": 0.04,
#     "prevention_rehab": 0.04,
#     "general": 0.08,
# }
DEFAULT_BUCKET_WEIGHTS = {
    "treatment": 0.28,
    "diagnosis": 0.22,
    "medical_knowledge": 0.40,
    "drug": 0.12,
    "exam_style": 0.30,
    "check_report": 0.06,
    "triage": 0.20,
    "prevention_rehab": 0.03,
    "general": 0.20,
}

MEDICAL_HELPFUL_KEYWORDS = [
    "治疗", "用药", "诊断", "鉴别", "原因", "病因", "机制", "检查", "指标",
    "建议", "注意", "护理", "预防", "康复", "副作用", "禁忌", "复查", "风险",
]

EXAM_STYLE_PATTERNS = [
    r"哪项", r"不属于", r"首选", r"最可能", r"主要表现", r"治疗原则", r"机制",
    r"病因", r"诊断依据", r"鉴别诊断", r"下列.*正确", r"下列.*错误", r"\bA[\.．、]",
    r"\bB[\.．、]", r"\bC[\.．、]", r"\bD[\.．、]",
]

DISCLAIMER_PATTERNS = [
    r"不能替代.*医生", r"仅供参考", r"建议及时就医", r"具体需结合.*情况", r"遵医嘱",
]

NOISE_PATTERNS = [
    r"作为.*AI", r"我无法.*诊断"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select a large subset from HuatuoGPT2-SFT-GPT4-140K.")
    p.add_argument("--dataset", default="FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K")
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--stats-output", default="")
    p.add_argument("--target-size", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)

    # Looser defaults than v1
    p.add_argument("--min-question-chars", type=int, default=2)
    p.add_argument("--min-answer-chars", type=int, default=20)
    p.add_argument("--max-answer-chars", type=int, default=2200)
    p.add_argument("--max-total-chars", type=int, default=3200)

    p.add_argument("--selection-mode", choices=["balanced", "global"], default="balanced")
    p.add_argument("--bucket-weights", default="")
    p.add_argument("--keep-bucket-metadata", action="store_true")
    p.add_argument("--dedupe-mode", choices=["qa", "conversation"], default="qa")
    return p.parse_args()


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compact_text(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def get_role(msg: Dict[str, Any]) -> str:
    return str(msg.get("from") or msg.get("role") or "").strip().lower()


def get_value(msg: Dict[str, Any]) -> str:
    value = msg.get("value")
    if value is None:
        value = msg.get("content", "")
    return str(value).strip()


def extract_qa(sample: Dict[str, Any]) -> Optional[Tuple[str, str, List[Dict[str, Any]]]]:
    convs = sample.get("conversations") or sample.get("messages")
    if not isinstance(convs, list) or not convs:
        return None
    if get_role(convs[-1]) not in ASSISTANT_ROLES:
        return None

    question = ""
    answer = ""
    for msg in convs:
        role = get_role(msg)
        value = get_value(msg)
        if role in USER_ROLES and value:
            question = value
        elif role in ASSISTANT_ROLES and value:
            answer = value

    if not question or not answer:
        return None
    return question, answer, convs


def infer_bucket(question: str, answer: str) -> str:
    both = question + "\n" + answer

    if re.search(r"急诊|胸痛|呼吸困难|昏迷|抽搐|大出血|高热|意识障碍|休克", both):
        return "triage"
    if re.search(r"药|用法|剂量|副作用|禁忌|联合用药|孕妇|哺乳期|儿童", both):
        return "drug"
    if re.search(r"化验|检查|指标|报告|影像|片子|B超|CT|MRI|血常规|肝功能|肾功能", both):
        return "check_report"
    if re.search(r"治疗|怎么治|如何治|疗法|手术|干预|方案", both):
        return "treatment"
    if re.search(r"诊断|鉴别|是什么病|考虑什么|最可能|诊断依据", both):
        return "diagnosis"
    if re.search(r"预防|护理|康复|饮食|生活方式|复诊|随访", both):
        return "prevention_rehab"
    if any(re.search(p, question) for p in EXAM_STYLE_PATTERNS):
        return "exam_style"
    if re.search(r"原因|病因|机制|表现|症状|并发症|传播|定义|属于|不属于", both):
        return "medical_knowledge"
    return "general"


def score_sample(question: str, answer: str, bucket: str, min_answer_chars: int, max_answer_chars: int, max_total_chars: int) -> float:
    q_len = len(question)
    a_len = len(answer)
    total_len = q_len + a_len

    if q_len < 2 or a_len < min_answer_chars:
        return -1e9
    if a_len > max_answer_chars * 1.6:
        return -1e9
    if total_len > max_total_chars * 1.7:
        return -1e9

    score = 0.0

    if 4 <= q_len <= 160:
        score += 0.8
    elif q_len <= 260:
        score += 0.3
    else:
        score -= 0.4

    if 40 <= a_len <= 1200:
        score += 1.8
    elif 20 <= a_len <= 1800:
        score += 1.0
    elif a_len <= max_answer_chars:
        score += 0.2
    else:
        score -= 0.6

    if total_len <= max_total_chars:
        score += 0.8
    else:
        score -= 0.4

    helpful_hits = sum(1 for kw in MEDICAL_HELPFUL_KEYWORDS if kw in answer or kw in question)
    score += min(helpful_hits, 6) * 0.18

    if bucket in {"treatment", "diagnosis", "medical_knowledge", "drug", "exam_style"}:
        score += 0.6

    if any(re.search(p, question) for p in EXAM_STYLE_PATTERNS):
        score += 0.8

    disclaimer_hits = sum(1 for p in DISCLAIMER_PATTERNS if re.search(p, answer))
    score -= min(disclaimer_hits, 3) * 0.1

    noise_hits = sum(1 for p in NOISE_PATTERNS if re.search(p, answer))
    score -= noise_hits * 1.0

    if a_len > 1500:
        score -= 0.4
    if a_len > 2000:
        score -= 0.6

    return score


def parse_bucket_weights(s: str) -> Dict[str, float]:
    if not s.strip():
        return dict(DEFAULT_BUCKET_WEIGHTS)
    out = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, v = chunk.split("=")
        out[k.strip()] = float(v.strip())
    total = sum(out.values())
    if total <= 0:
        raise ValueError("bucket weights sum must be > 0")
    return {k: v / total for k, v in out.items()}


def main() -> None:
    args = parse_args()
    bucket_weights = parse_bucket_weights(args.bucket_weights)

    # ds = load_dataset(args.dataset, split=args.split)
    ds = load_dataset("json", data_files=args.dataset, split=args.split)

    seen = set()
    rows_all: List[Dict[str, Any]] = []
    rows_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    invalid = 0
    too_short = 0
    too_long = 0
    deduped = 0

    for sample in ds:
        qa = extract_qa(sample)
        if qa is None:
            invalid += 1
            continue

        question, answer, convs = qa
        q_len = len(question)
        a_len = len(answer)
        total_len = q_len + a_len

        if q_len < args.min_question_chars or a_len < args.min_answer_chars:
            too_short += 1
            continue
        if a_len > args.max_answer_chars or total_len > args.max_total_chars:
            too_long += 1
            continue

        if args.dedupe_mode == "qa":
            dedupe_key = sha1_text(compact_text(question) + "||" + compact_text(answer))
        else:
            dedupe_key = sha1_text(compact_text(json.dumps(convs, ensure_ascii=False)))

        if dedupe_key in seen:
            deduped += 1
            continue
        seen.add(dedupe_key)

        bucket = infer_bucket(question, answer)
        score = score_sample(
            question=question,
            answer=answer,
            bucket=bucket,
            min_answer_chars=args.min_answer_chars,
            max_answer_chars=args.max_answer_chars,
            max_total_chars=args.max_total_chars,
        )
        if score <= -1e8:
            continue

        row = {
            "id": sample.get("id", ""),
            "bucket": bucket,
            "score": float(score),
            "question_chars": q_len,
            "answer_chars": a_len,
            "total_chars": total_len,
            "conversations": convs,
        }
        rows_all.append(row)
        rows_by_bucket[bucket].append(row)

    for bucket in rows_by_bucket:
        rows_by_bucket[bucket].sort(key=lambda x: (-x["score"], x["answer_chars"], x["question_chars"]))

    rows_all.sort(key=lambda x: (-x["score"], x["answer_chars"], x["question_chars"]))

    target_size = min(args.target_size, len(rows_all))
    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    if args.selection_mode == "balanced":
        quotas = {bucket: int(round(target_size * weight)) for bucket, weight in bucket_weights.items()}
        for bucket, quota in quotas.items():
            for row in rows_by_bucket.get(bucket, [])[:quota]:
                key = sha1_text(json.dumps(row["conversations"], ensure_ascii=False, sort_keys=True))
                if key in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(key)

        # Global fill to guarantee large target size
        for row in rows_all:
            if len(selected) >= target_size:
                break
            key = sha1_text(json.dumps(row["conversations"], ensure_ascii=False, sort_keys=True))
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)

    else:
        for row in rows_all:
            if len(selected) >= target_size:
                break
            key = sha1_text(json.dumps(row["conversations"], ensure_ascii=False, sort_keys=True))
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(selected):
            out = {
                "id": row["id"] if row["id"] else f"huatuo_sel_{i}",
                "source": "FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K",
                "conversations": row["conversations"],
            }
            if args.keep_bucket_metadata:
                out["bucket"] = row["bucket"]
                out["selection_score"] = round(float(row["score"]), 4)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    bucket_counter = Counter(row["bucket"] for row in selected)
    score_vals = [row["score"] for row in selected]
    answer_lens = [row["answer_chars"] for row in selected]
    question_lens = [row["question_chars"] for row in selected]
    total_lens = [row["total_chars"] for row in selected]

    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "invalid_rows_skipped": invalid,
        "too_short_skipped": too_short,
        "too_long_skipped": too_long,
        "deduped_skipped": deduped,
        "candidate_rows_after_filter": len(rows_all),
        "target_size_requested": args.target_size,
        "selected_size": len(selected),
        "selection_mode": args.selection_mode,
        "bucket_distribution": dict(bucket_counter),
        "score_mean": round(sum(score_vals) / max(1, len(score_vals)), 4) if score_vals else None,
        "score_min": round(min(score_vals), 4) if score_vals else None,
        "score_max": round(max(score_vals), 4) if score_vals else None,
        "answer_chars_mean": round(sum(answer_lens) / max(1, len(answer_lens)), 2) if answer_lens else None,
        "question_chars_mean": round(sum(question_lens) / max(1, len(question_lens)), 2) if question_lens else None,
        "total_chars_mean": round(sum(total_lens) / max(1, len(total_lens)), 2) if total_lens else None,
        "bucket_weights": bucket_weights,
        "output": str(out_path),
    }

    stats_path = Path(args.stats_output) if args.stats_output else out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
