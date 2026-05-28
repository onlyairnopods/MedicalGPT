#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze prototype-retrieved ShareGPT training data.

Input:
- selected_sharegpt.jsonl
  Expected fields (recommended):
    id
    parent_conversation_id
    source
    prototype_id
    prototype_task_type
    prototype_bucket
    prototype_score
    conversations
    tools (optional)

Outputs:
- summary.json
- prototype_stats.csv / .json
- source_stats.csv
- task_type_stats.csv
- quality_flags.csv
- recommendations.md
- filter_summary.json
- filtered_selected_sharegpt.jsonl (optional)
- dropped_records.jsonl (optional)

Typical usage:
python analyze_retrieved_data.py \
  --input selected_sharegpt.jsonl \
  --output-dir analysis_out \
  --write-filtered \
  --min-score 0.55 \
  --min-assistant-chars 20 \
  --max-total-chars 4000 \
  --dedupe normalized \
  --per-prototype-topk 500 \
  --per-prototype-min-quantile 0.1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "assistant"}
TOOL_ROLES = {"function_call", "function", "tool", "observation"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze prototype-retrieved ShareGPT data.")
    p.add_argument("--input", required=True, help="selected_sharegpt.jsonl")
    p.add_argument("--output-dir", required=True, help="Directory for reports")
    p.add_argument("--write-filtered", action="store_true", help="Write filtered_selected_sharegpt.jsonl and dropped_records.jsonl")
    p.add_argument("--min-score", type=float, default=-1.0, help="Drop rows with prototype_score below this. <0 disables")
    p.add_argument("--min-assistant-chars", type=int, default=0, help="Drop if last assistant answer shorter than this")
    p.add_argument("--max-total-chars", type=int, default=0, help="Drop if total chars exceed this. 0 disables")
    p.add_argument("--dedupe", choices=["none", "exact", "normalized"], default="normalized", help="Deduplication mode")
    p.add_argument("--per-prototype-topk", type=int, default=0, help="Keep only top-k by prototype_score within each prototype. 0 disables")
    p.add_argument("--per-prototype-min-quantile", type=float, default=-1.0, help="Within each prototype, drop bottom score quantile. Example 0.1 means drop bottom 10 percent")
    p.add_argument("--sample-bad-cases", type=int, default=200, help="How many dropped / flagged cases to keep in dropped_records.jsonl")
    return p.parse_args()


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc


def role_of(msg: Dict[str, Any]) -> str:
    return str(msg.get("from") or msg.get("role") or "").strip().lower()


def value_of(msg: Dict[str, Any]) -> str:
    value = msg.get("value")
    if value is None:
        value = msg.get("content", "")
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compact_text(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class RowStats:
    row_idx: int
    id: str
    parent_conversation_id: str
    source: str
    prototype_id: str
    prototype_task_type: str
    prototype_bucket: str
    prototype_score: Optional[float]
    num_turns: int
    num_user_turns: int
    num_assistant_turns: int
    num_tool_turns: int
    total_chars: int
    last_assistant_chars: int
    has_tools: bool
    malformed: bool
    empty_last_assistant: bool
    too_short_answer: bool
    too_long_total: bool
    exact_dup_key: str
    normalized_dup_key: str


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def last_assistant_answer(conversations: List[Dict[str, Any]]) -> str:
    for msg in reversed(conversations):
        if role_of(msg) in ASSISTANT_ROLES:
            return value_of(msg)
    return ""


def build_row_stats(row: Dict[str, Any], row_idx: int, min_assistant_chars: int, max_total_chars: int) -> RowStats:
    convs = row.get("conversations") or row.get("messages") or []
    if not isinstance(convs, list):
        convs = []

    roles = [role_of(m) for m in convs]
    vals = [value_of(m) for m in convs]
    total_chars = sum(len(v) for v in vals)
    last_answer = last_assistant_answer(convs)
    last_answer_chars = len(last_answer)

    malformed = False
    if not convs:
        malformed = True
    else:
        if roles[-1] not in ASSISTANT_ROLES:
            malformed = True
        if sum(1 for r in roles if r in ASSISTANT_ROLES) == 0:
            malformed = True

    exact_base = json.dumps(convs, ensure_ascii=False, sort_keys=True)
    normalized_base = "\n".join(f"{role_of(m)}:{compact_text(value_of(m))}" for m in convs)

    return RowStats(
        row_idx=row_idx,
        id=str(row.get("id", "")),
        parent_conversation_id=str(row.get("parent_conversation_id", "")),
        source=str(row.get("source", "unknown")),
        prototype_id=str(row.get("prototype_id", "")),
        prototype_task_type=str(row.get("prototype_task_type", "")),
        prototype_bucket=str(row.get("prototype_bucket", "")),
        prototype_score=safe_float(row.get("prototype_score")),
        num_turns=len(convs),
        num_user_turns=sum(1 for r in roles if r in USER_ROLES),
        num_assistant_turns=sum(1 for r in roles if r in ASSISTANT_ROLES),
        num_tool_turns=sum(1 for r in roles if r in TOOL_ROLES),
        total_chars=total_chars,
        last_assistant_chars=last_answer_chars,
        has_tools=("tools" in row) or any(r in TOOL_ROLES for r in roles),
        malformed=malformed,
        empty_last_assistant=(last_answer_chars == 0),
        too_short_answer=(min_assistant_chars > 0 and last_answer_chars < min_assistant_chars),
        too_long_total=(max_total_chars > 0 and total_chars > max_total_chars),
        exact_dup_key=sha1(exact_base),
        normalized_dup_key=sha1(normalized_base),
    )


def mean_or_none(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def median_or_none(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    if q <= 0:
        return float(min(xs))
    if q >= 1:
        return float(max(xs))
    arr = sorted(xs)
    idx = (len(arr) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(arr[lo])
    frac = idx - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def bucket_counts(values: List[int], bins: List[Tuple[str, int, int]]) -> Dict[str, int]:
    out = {name: 0 for name, _, _ in bins}
    for v in values:
        for name, lo, hi in bins:
            if lo <= v <= hi:
                out[name] += 1
                break
    return out


def analyze(rows: List[Dict[str, Any]], args: argparse.Namespace, output_dir: Path):
    stats: List[RowStats] = [build_row_stats(r, i, args.min_assistant_chars, args.max_total_chars) for i, r in enumerate(rows)]

    n = len(stats)
    score_vals = [s.prototype_score for s in stats if s.prototype_score is not None]
    total_chars_vals = [s.total_chars for s in stats]
    answer_chars_vals = [s.last_assistant_chars for s in stats]

    exact_counter = Counter(s.exact_dup_key for s in stats)
    norm_counter = Counter(s.normalized_dup_key for s in stats)

    by_proto: Dict[str, List[RowStats]] = defaultdict(list)
    by_source: Dict[str, List[RowStats]] = defaultdict(list)
    by_task: Dict[str, List[RowStats]] = defaultdict(list)
    by_bucket: Dict[str, List[RowStats]] = defaultdict(list)

    for s in stats:
        by_proto[s.prototype_id].append(s)
        by_source[s.source].append(s)
        by_task[s.prototype_task_type].append(s)
        by_bucket[s.prototype_bucket].append(s)

    prototype_rows: List[Dict[str, Any]] = []
    for proto_id, items in sorted(by_proto.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        scores = [x.prototype_score for x in items if x.prototype_score is not None]
        prototype_rows.append({
            "prototype_id": proto_id,
            "task_type": items[0].prototype_task_type if items else "",
            "count": len(items),
            "share": round(len(items) / max(1, n), 6),
            "score_mean": mean_or_none(scores),
            "score_median": median_or_none(scores),
            "score_p10": quantile(scores, 0.1),
            "score_p90": quantile(scores, 0.9),
            "assistant_chars_mean": mean_or_none([x.last_assistant_chars for x in items]),
            "total_chars_mean": mean_or_none([x.total_chars for x in items]),
            "has_tools_rate": round(sum(1 for x in items if x.has_tools) / max(1, len(items)), 6),
            "malformed_rate": round(sum(1 for x in items if x.malformed) / max(1, len(items)), 6),
            "dup_norm_rate": round(sum(1 for x in items if norm_counter[x.normalized_dup_key] > 1) / max(1, len(items)), 6),
            "sources": "|".join(f"{k}:{v}" for k, v in Counter(x.source for x in items).most_common(5)),
            "buckets": "|".join(f"{k}:{v}" for k, v in Counter(x.prototype_bucket for x in items).most_common()),
        })

    source_rows = []
    for source, items in sorted(by_source.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        scores = [x.prototype_score for x in items if x.prototype_score is not None]
        source_rows.append({
            "source": source,
            "count": len(items),
            "share": round(len(items) / max(1, n), 6),
            "score_mean": mean_or_none(scores),
            "assistant_chars_mean": mean_or_none([x.last_assistant_chars for x in items]),
            "malformed_rate": round(sum(1 for x in items if x.malformed) / max(1, len(items)), 6),
        })

    task_rows = []
    for task, items in sorted(by_task.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        scores = [x.prototype_score for x in items if x.prototype_score is not None]
        task_rows.append({
            "prototype_task_type": task,
            "count": len(items),
            "share": round(len(items) / max(1, n), 6),
            "score_mean": mean_or_none(scores),
            "assistant_chars_mean": mean_or_none([x.last_assistant_chars for x in items]),
            "has_tools_rate": round(sum(1 for x in items if x.has_tools) / max(1, len(items)), 6),
        })

    quality_rows: List[Dict[str, Any]] = []
    for s in stats:
        flags = []
        if s.malformed:
            flags.append("malformed")
        if s.empty_last_assistant:
            flags.append("empty_last_assistant")
        if s.too_short_answer:
            flags.append("too_short_answer")
        if s.too_long_total:
            flags.append("too_long_total")
        if exact_counter[s.exact_dup_key] > 1:
            flags.append("exact_duplicate")
        if norm_counter[s.normalized_dup_key] > 1:
            flags.append("normalized_duplicate")
        if (s.prototype_score is None) or (args.min_score >= 0 and s.prototype_score is not None and s.prototype_score < args.min_score):
            flags.append("low_or_missing_score")
        if flags:
            quality_rows.append({
                "row_idx": s.row_idx,
                "id": s.id,
                "prototype_id": s.prototype_id,
                "prototype_task_type": s.prototype_task_type,
                "source": s.source,
                "prototype_bucket": s.prototype_bucket,
                "prototype_score": s.prototype_score,
                "num_turns": s.num_turns,
                "last_assistant_chars": s.last_assistant_chars,
                "total_chars": s.total_chars,
                "flags": "|".join(flags),
            })

    turn_bins = [
        ("1-2", 1, 2),
        ("3-4", 3, 4),
        ("5-6", 5, 6),
        ("7-10", 7, 10),
        ("11+", 11, 10**9),
    ]
    ans_bins = [
        ("0", 0, 0),
        ("1-20", 1, 20),
        ("21-80", 21, 80),
        ("81-200", 81, 200),
        ("201-500", 201, 500),
        ("501+", 501, 10**9),
    ]
    total_bins = [
        ("0-200", 0, 200),
        ("201-800", 201, 800),
        ("801-1600", 801, 1600),
        ("1601-3200", 1601, 3200),
        ("3201+", 3201, 10**9),
    ]

    summary = {
        "input_rows": n,
        "unique_parent_conversation_ids": len(set(s.parent_conversation_id for s in stats)),
        "unique_prototypes_hit": len(by_proto),
        "unique_task_types_hit": len(by_task),
        "unique_sources": len(by_source),
        "with_tools_count": sum(1 for s in stats if s.has_tools),
        "with_tools_rate": round(sum(1 for s in stats if s.has_tools) / max(1, n), 6),
        "malformed_count": sum(1 for s in stats if s.malformed),
        "malformed_rate": round(sum(1 for s in stats if s.malformed) / max(1, n), 6),
        "empty_last_assistant_count": sum(1 for s in stats if s.empty_last_assistant),
        "too_short_answer_count": sum(1 for s in stats if s.too_short_answer),
        "too_long_total_count": sum(1 for s in stats if s.too_long_total),
        "exact_duplicate_rows": sum(1 for s in stats if exact_counter[s.exact_dup_key] > 1),
        "normalized_duplicate_rows": sum(1 for s in stats if norm_counter[s.normalized_dup_key] > 1),
        "prototype_score": {
            "count": len(score_vals),
            "mean": mean_or_none(score_vals),
            "median": median_or_none(score_vals),
            "p10": quantile(score_vals, 0.1),
            "p25": quantile(score_vals, 0.25),
            "p75": quantile(score_vals, 0.75),
            "p90": quantile(score_vals, 0.9),
            "min": min(score_vals) if score_vals else None,
            "max": max(score_vals) if score_vals else None,
        },
        "last_assistant_chars": {
            "mean": mean_or_none(answer_chars_vals),
            "median": median_or_none(answer_chars_vals),
            "p10": quantile(answer_chars_vals, 0.1),
            "p90": quantile(answer_chars_vals, 0.9),
            "distribution": bucket_counts(answer_chars_vals, ans_bins),
        },
        "total_chars": {
            "mean": mean_or_none(total_chars_vals),
            "median": median_or_none(total_chars_vals),
            "p10": quantile(total_chars_vals, 0.1),
            "p90": quantile(total_chars_vals, 0.9),
            "distribution": bucket_counts(total_chars_vals, total_bins),
        },
        "num_turns": {
            "mean": mean_or_none([s.num_turns for s in stats]),
            "median": median_or_none([s.num_turns for s in stats]),
            "distribution": bucket_counts([s.num_turns for s in stats], turn_bins),
        },
        "bucket_distribution": dict(Counter(s.prototype_bucket for s in stats)),
        "top_prototypes": prototype_rows[:20],
        "top_sources": source_rows[:20],
        "top_task_types": task_rows[:20],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "prototype_stats.json").write_text(json.dumps(prototype_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "prototype_stats.csv", prototype_rows, list(prototype_rows[0].keys()) if prototype_rows else ["prototype_id", "count"])
    write_csv(output_dir / "source_stats.csv", source_rows, list(source_rows[0].keys()) if source_rows else ["source", "count"])
    write_csv(output_dir / "task_type_stats.csv", task_rows, list(task_rows[0].keys()) if task_rows else ["prototype_task_type", "count"])
    write_csv(output_dir / "quality_flags.csv", quality_rows, list(quality_rows[0].keys()) if quality_rows else ["row_idx", "id", "flags"])

    keep = [True] * n
    drop_reasons: Dict[int, List[str]] = defaultdict(list)

    for i, s in enumerate(stats):
        if s.malformed:
            keep[i] = False
            drop_reasons[i].append("malformed")
        if args.min_score >= 0 and (s.prototype_score is None or s.prototype_score < args.min_score):
            keep[i] = False
            drop_reasons[i].append("below_min_score")
        if args.min_assistant_chars > 0 and s.last_assistant_chars < args.min_assistant_chars:
            keep[i] = False
            drop_reasons[i].append("below_min_assistant_chars")
        if args.max_total_chars > 0 and s.total_chars > args.max_total_chars:
            keep[i] = False
            drop_reasons[i].append("above_max_total_chars")

    if args.dedupe != "none":
        seen = set()
        for i, s in enumerate(stats):
            key = s.exact_dup_key if args.dedupe == "exact" else s.normalized_dup_key
            if key in seen:
                keep[i] = False
                drop_reasons[i].append(f"{args.dedupe}_duplicate")
            else:
                seen.add(key)

    if args.per_prototype_min_quantile >= 0:
        proto_score_threshold: Dict[str, float] = {}
        for proto_id, items in by_proto.items():
            scores = [x.prototype_score for x in items if x.prototype_score is not None]
            qv = quantile(scores, args.per_prototype_min_quantile)
            if qv is not None:
                proto_score_threshold[proto_id] = qv
        for i, s in enumerate(stats):
            thr = proto_score_threshold.get(s.prototype_id)
            if thr is not None and s.prototype_score is not None and s.prototype_score < thr:
                keep[i] = False
                drop_reasons[i].append(f"below_prototype_quantile_{args.per_prototype_min_quantile}")

    if args.per_prototype_topk > 0:
        proto_rows_sorted: Dict[str, List[Tuple[int, RowStats]]] = defaultdict(list)
        for i, s in enumerate(stats):
            proto_rows_sorted[s.prototype_id].append((i, s))
        for proto_id, arr in proto_rows_sorted.items():
            arr.sort(key=lambda t: (t[1].prototype_score is None, -(t[1].prototype_score or -1e9), t[1].row_idx))
            for i, _ in arr[args.per_prototype_topk:]:
                keep[i] = False
                drop_reasons[i].append("beyond_per_prototype_topk")

    kept_rows = [rows[i] for i in range(n) if keep[i]]
    dropped_examples = []
    for i in range(n):
        if not keep[i]:
            item = dict(rows[i])
            item["_drop_reasons"] = drop_reasons[i]
            dropped_examples.append(item)

    filter_summary = {
        "input_rows": n,
        "kept_rows": len(kept_rows),
        "dropped_rows": n - len(kept_rows),
        "kept_rate": round(len(kept_rows) / max(1, n), 6),
        "drop_reason_counts": dict(Counter(r for rs in drop_reasons.values() for r in rs)),
        "settings": {
            "min_score": args.min_score,
            "min_assistant_chars": args.min_assistant_chars,
            "max_total_chars": args.max_total_chars,
            "dedupe": args.dedupe,
            "per_prototype_topk": args.per_prototype_topk,
            "per_prototype_min_quantile": args.per_prototype_min_quantile,
        },
    }
    (output_dir / "filter_summary.json").write_text(json.dumps(filter_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rec_lines = []
    rec_lines.append("# Retrieved Data Analysis Recommendations\n")
    rec_lines.append(f"- 总样本数：**{n}**")
    rec_lines.append(f"- 命中的 prototype 数：**{len(by_proto)}**")
    rec_lines.append(f"- 规范化重复样本数：**{summary['normalized_duplicate_rows']}**")
    rec_lines.append(f"- malformed 比例：**{summary['malformed_rate']:.2%}**")
    if score_vals:
        rec_lines.append(f"- prototype_score 中位数：**{summary['prototype_score']['median']:.4f}**")
        rec_lines.append(f"- prototype_score P10 / P90：**{summary['prototype_score']['p10']:.4f} / {summary['prototype_score']['p90']:.4f}**")
    rec_lines.append("\n## 过滤建议\n")
    rec_lines.append("- 先做 **normalized dedupe**，去除模板化和近重复窗口。")
    rec_lines.append("- 对 `prototype_score` 做全局下限，起点可先试 **0.50 ~ 0.65**。")
    rec_lines.append("- 再做 `per_prototype_min_quantile=0.1`，去掉每个 prototype 内部分数最低的 10%。")
    rec_lines.append("- 若头部 prototype 过多，可加 `per_prototype_topk` 控制单类上限。")
    rec_lines.append("- 对 `last_assistant_chars` 太短的样本直接丢弃，起点可先试 **20**。")
    rec_lines.append("- 对超长样本做上限，避免把冗长百科型回答全吃进去。")
    (output_dir / "recommendations.md").write_text("\n".join(rec_lines), encoding="utf-8")

    if args.write_filtered:
        filter_file_name = args.input.replace(".jsonl", "_filtered.jsonl")
        with (output_dir / filter_file_name).open("w", encoding="utf-8") as f:
            for r in kept_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dropped_file_name = args.input.replace(".jsonl", "_dropped_records.jsonl")
        with (output_dir / dropped_file_name).open("w", encoding="utf-8") as f:
            for r in dropped_examples[:args.sample_bad_cases]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "summary": summary,
        "filter_summary": filter_summary,
        "prototype_stats_rows": len(prototype_rows),
        "quality_flags_rows": len(quality_rows),
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = list(load_jsonl(input_path))
    result = analyze(rows, args, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
