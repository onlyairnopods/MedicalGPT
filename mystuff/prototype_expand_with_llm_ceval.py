#!/usr/bin/env python3
"""Refine and expand prototype.jsonl using CEval medical subsets + OpenAI Responses API.

This version reads CEval directly from Hugging Face datasets and only uses 3 subsets:
  - basic_medicine (val)
  - clinical_medicine (val)
  - physician (val)

What it does:
  1. Loads existing prototypes.
  2. Loads the 3 CEval medical subsets from HF datasets.
  3. Converts each MCQ row into a compact QA-style example.
  4. Samples a small batch (or all examples).
  5. Sends current prototypes + benchmark examples to an LLM.
  6. Gets structured JSON back: updated prototypes + newly proposed prototypes.
  7. Merges/deduplicates and writes a new prototype jsonl.

Requirements:
  pip install openai datasets
  export OPENAI_API_KEY=...

Example:
  python prototype_expand_with_llm_ceval.py \
      --prototypes prototype.jsonl \
      --output prototype_expanded.jsonl \
      --proposal-output prototype_proposals.json \
      --model gpt-4o-mini \
      --sample-size 90 \
      --batch-size 18
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset
from openai import OpenAI

ALLOWED_RISK = {"low", "medium", "high"}
CEVAL_SUBSETS = ["basic_medicine", "clinical_medicine", "physician"]
CEVAL_SPLIT = "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototypes", required=True, help="existing prototype.jsonl")
    parser.add_argument("--output", required=True, help="new prototype jsonl")
    parser.add_argument("--proposal-output", default="", help="optional raw proposal json")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model")
    parser.add_argument("--sample-size", type=int, default=90, help="sample this many QA examples before batching; 0 means all")
    parser.add_argument("--batch-size", type=int, default=8, help="examples per LLM call")
    parser.add_argument("--max-new-per-batch", type=int, default=6)
    parser.add_argument("--max-update-per-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-existing-in-prompt", type=int, default=200, help="existing prototypes to include in prompt")
    return parser.parse_args()


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return rows

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_prototypes(path: str) -> List[Dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    cleaned = []
    for row in rows:
        cleaned.append(
            {
                "prototype_id": str(row["prototype_id"]),
                "task_type": str(row.get("task_type", "general")),
                "description": str(row["description"]).strip(),
                "need_tool": bool(row.get("need_tool", False)),
                "risk_level": normalize_risk(row.get("risk_level", "medium")),
            }
        )
    return cleaned


def normalize_risk(value: Any) -> str:
    v = str(value).strip().lower()
    if v not in ALLOWED_RISK:
        return "medium"
    return v


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def ceval_row_to_example(record: Dict[str, Any], subset_name: str, idx: int) -> Dict[str, str]:
    question = normalize_text(str(record.get("question", "")))
    if not question:
        raise ValueError(f"Empty question in subset={subset_name}, idx={idx}")

    options = []
    option_map: Dict[str, str] = {}
    for key in ["A", "B", "C", "D"]:
        value = normalize_text(str(record.get(key, "")))
        if value:
            option_map[key] = value
            options.append(f"{key}. {value}")

    answer_letter = normalize_text(str(record.get("answer", ""))).upper()
    answer_text = option_map.get(answer_letter, "")
    explanation = normalize_text(str(record.get("explanation", "")))

    composed_question = question
    if options:
        composed_question += "\n选项:\n" + "\n".join(options)

    answer_parts = []
    if answer_letter:
        if answer_text:
            answer_parts.append(f"正确答案: {answer_letter}. {answer_text}")
        else:
            answer_parts.append(f"正确答案: {answer_letter}")
    if explanation:
        answer_parts.append(f"解析: {explanation}")

    return {
        "example_id": f"{subset_name}_{record.get('id', idx)}",
        "subset": subset_name,
        "question": composed_question,
        "answer": "\n".join(answer_parts),
    }



def load_ceval_medical_examples() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for subset_name in CEVAL_SUBSETS:
        ds = load_dataset("ceval/ceval-exam", subset_name, split=CEVAL_SPLIT)
        for idx, record in enumerate(ds):
            rows.append(ceval_row_to_example(record, subset_name=subset_name, idx=idx))
    return rows



def sample_examples(rows: List[Dict[str, str]], sample_size: int, seed: int) -> List[Dict[str, str]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return rows

    by_subset: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_subset.setdefault(row.get("subset", "unknown"), []).append(row)

    rng = random.Random(seed)
    subset_names = sorted(by_subset.keys())
    sampled: List[Dict[str, str]] = []

    # roughly balanced across the 3 CEval subsets
    base = sample_size // max(1, len(subset_names))
    remainder = sample_size % max(1, len(subset_names))

    for i, subset in enumerate(subset_names):
        take = base + (1 if i < remainder else 0)
        pool = by_subset[subset][:]
        rng.shuffle(pool)
        sampled.extend(pool[: min(take, len(pool))])

    # fill any shortage from the global pool
    if len(sampled) < sample_size:
        picked_ids = {x["example_id"] for x in sampled}
        leftovers = [x for x in rows if x["example_id"] not in picked_ids]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: sample_size - len(sampled)])

    rng.shuffle(sampled)
    return sampled



def batchify(rows: Sequence[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    return [list(rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)]



def compact_prototypes_for_prompt(prototypes: Sequence[Dict[str, Any]], max_existing: int) -> List[Dict[str, Any]]:
    rows = list(prototypes)[:max_existing]
    return [
        {
            "prototype_id": p["prototype_id"],
            "task_type": p["task_type"],
            "description": p["description"],
            "need_tool": p["need_tool"],
            "risk_level": p["risk_level"],
        }
        for p in rows
    ]



def make_proposal_schema() -> Dict[str, Any]:
    proto_obj = {
        "type": "object",
        "properties": {
            "prototype_id": {"type": "string"},
            "task_type": {"type": "string"},
            "description": {"type": "string"},
            "need_tool": {"type": "boolean"},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["prototype_id", "task_type", "description", "need_tool", "risk_level"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "insights": {"type": "array", "items": {"type": "string"}},
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_prototype_id": {"type": "string"},
                        "prototype": proto_obj,
                        "reason": {"type": "string"},
                    },
                    "required": ["target_prototype_id", "prototype", "reason"],
                    "additionalProperties": False,
                },
            },
            "new_prototypes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prototype": proto_obj,
                        "reason": {"type": "string"},
                    },
                    "required": ["prototype", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["insights", "updates", "new_prototypes"],
        "additionalProperties": False,
    }



def make_merge_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prototypes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prototype_id": {"type": "string"},
                        "task_type": {"type": "string"},
                        "description": {"type": "string"},
                        "need_tool": {"type": "boolean"},
                        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["prototype_id", "task_type", "description", "need_tool", "risk_level"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["prototypes"],
        "additionalProperties": False,
    }



def call_structured_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    )
    return json.loads(response.output_text)



def build_batch_prompt(
    existing_prototypes: Sequence[Dict[str, Any]],
    batch_examples: Sequence[Dict[str, str]],
    max_new_per_batch: int,
    max_update_per_batch: int,
) -> Tuple[str, str]:
    system_prompt = (
        "你是一个医疗SFT数据设计助手。你的任务不是回答医疗问题，"
        "而是根据一批CEval医学Benchmark样本，改进现有prototype库。"
        "请尽量做两件事：1）修正现有prototype描述，使其更覆盖真实任务；"
        "2）在确有必要时新增prototype。"
        "输出必须是JSON，不要输出任何额外解释。"
    )

    user_prompt = (
        "下面是当前prototype库（精简版）:\n"
        f"{json.dumps(existing_prototypes, ensure_ascii=False, indent=2)}\n\n"
        "下面是一批CEval医学Benchmark样本（来自 basic_medicine / clinical_medicine / physician）:\n"
        f"{json.dumps(batch_examples, ensure_ascii=False, indent=2)}\n\n"
        f"请基于这些样本，最多提出 {max_update_per_batch} 个更新、{max_new_per_batch} 个新增。\n"
        "规则：\n"
        "1. 优先复用现有task_type，除非确实出现新的任务类型。\n"
        "2. description必须是抽象能力描述，不要抄具体题目原文。\n"
        "3. prototype_id新增时用短ID，例如 report_07、triage_09、tool_10。\n"
        "4. need_tool 只有在明显需要检索/计算/规则/工具调用时才设为 true。\n"
        "5. risk_level 只允许 low / medium / high。\n"
        "6. 如果某条现有prototype已经足够好，就不要更新它。\n"
        "7. 不要生成互相重复的prototype。\n"
        "8. 这些样本很多是医学考试题，不要把prototype写成“选择题作答”，而要抽象成背后的医疗能力，例如：概念辨析、机制解释、鉴别判断、检查解读、诊疗决策、药理知识等。\n"
    )
    return system_prompt, user_prompt



def build_merge_prompt(
    existing_prototypes: Sequence[Dict[str, Any]],
    batch_proposals: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    system_prompt = (
        "你是一个prototype库合并器。你的任务是把现有prototype和多批提案合并成最终版本。"
        "要求去重、统一风格、保留有价值的新增，并避免产生高度重叠的描述。"
        "输出必须是JSON。"
    )
    user_prompt = (
        "现有prototype:\n"
        f"{json.dumps(existing_prototypes, ensure_ascii=False, indent=2)}\n\n"
        "多批LLM提案:\n"
        f"{json.dumps(batch_proposals, ensure_ascii=False, indent=2)}\n\n"
        "请输出最终prototype列表。规则：\n"
        "1. 尽量保留原有prototype_id；新增prototype才创建新ID。\n"
        "2. 删除明显重复或只是在措辞上重复的prototype。\n"
        "3. description要抽象、稳定、适合后续做embedding召回。\n"
        "4. task_type尽量控制在少量清晰类别内，不要过度发散。\n"
        "5. risk_level 和 need_tool 保持一致且保守。\n"
        "6. 如果有明显偏向“考试答题形式”的描述，改写成通用医疗能力描述。\n"
    )
    return system_prompt, user_prompt



def normalize_proto(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prototype_id": normalize_text(str(p.get("prototype_id", ""))),
        "task_type": normalize_text(str(p.get("task_type", "general"))),
        "description": normalize_text(str(p.get("description", ""))),
        "need_tool": bool(p.get("need_tool", False)),
        "risk_level": normalize_risk(p.get("risk_level", "medium")),
    }



def dedupe_by_id_and_desc(prototypes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_id = set()
    seen_desc = set()
    out: List[Dict[str, Any]] = []
    for row in prototypes:
        p = normalize_proto(row)
        if not p["prototype_id"] or not p["description"]:
            continue
        desc_key = p["description"].lower()
        if p["prototype_id"] in seen_id:
            continue
        if desc_key in seen_desc:
            continue
        seen_id.add(p["prototype_id"])
        seen_desc.add(desc_key)
        out.append(p)
    return out



def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    prototypes = load_prototypes(args.prototypes)
    qa_rows = load_ceval_medical_examples()

    if not qa_rows:
        raise ValueError("No QA examples found in CEval subsets.")

    sampled = sample_examples(qa_rows, args.sample_size, args.seed)
    batches = batchify(sampled, args.batch_size)

    client = OpenAI(
        base_url="",
        api_key="",
    )
    existing_for_prompt = compact_prototypes_for_prompt(prototypes, args.max_existing_in_prompt)

    all_proposals: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(batches, 1):
        print(f"[proposal {batch_idx}/{len(batches)}] examples={len(batch)}")
        system_prompt, user_prompt = build_batch_prompt(
            existing_prototypes=existing_for_prompt,
            batch_examples=batch,
            max_new_per_batch=args.max_new_per_batch,
            max_update_per_batch=args.max_update_per_batch,
        )
        proposal = call_structured_json(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="prototype_batch_proposal",
            schema=make_proposal_schema(),
        )
        proposal["batch_index"] = batch_idx
        proposal["examples"] = batch
        all_proposals.append(proposal)

    if args.proposal_output:
        write_json(args.proposal_output, all_proposals)

    print("[merge] merging all proposals")
    merge_system, merge_user = build_merge_prompt(prototypes, all_proposals)
    merged = call_structured_json(
        client=client,
        model=args.model,
        system_prompt=merge_system,
        user_prompt=merge_user,
        schema_name="prototype_merge_result",
        schema=make_merge_schema(),
    )

    final_prototypes = dedupe_by_id_and_desc(merged["prototypes"])
    write_jsonl(args.output, final_prototypes)
    print(f"Saved final prototypes: {len(final_prototypes)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
