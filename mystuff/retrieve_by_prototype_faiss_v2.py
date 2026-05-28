#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
retrieve_by_prototype_faiss_v2.py

Compared with the earlier retrieve script:
- joins by metadata.record_id instead of conversation_id
- verifies window_hash before writing output
- skips mismatched rows and records them in warnings.jsonl
- writes prototype info directly into selected_sharegpt.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise SystemExit("Please install faiss-cpu or faiss-gpu first.") from exc


USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "assistant"}
TOOL_ROLES = {"function_call", "tool", "function", "observation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype retrieval with FAISS for ShareGPT candidate pools.")
    parser.add_argument("--input", required=True, help="Original ShareGPT .json/.jsonl file used to build embeddings.")
    parser.add_argument("--prototypes", required=True, help="prototype.jsonl")
    parser.add_argument("--emb-dir", required=True, help="Embedding cache dir containing metadata.jsonl and manifest.json")
    parser.add_argument("--index", required=True, help="FAISS index path")
    parser.add_argument("--output", required=True, help="Output selected ShareGPT sub-conversations (.jsonl)")
    parser.add_argument("--debug-output", default="", help="Optional debug JSONL with matched prototypes and scores")
    parser.add_argument("--warnings-output", default="")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B", help="SentenceTransformer model for prototype encoding. Default: manifest['model']")
    parser.add_argument("--device", default="cuda", help="Encoding device for prototypes, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=2048, help="Prototype encoding batch size")
    parser.add_argument("--normalize-queries", action="store_true", help="Force normalize prototype embeddings before search")
    parser.add_argument("--nprobe", type=int, default=32, help="FAISS nprobe for IVF indices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output")

    # selection params
    parser.add_argument("--top-high", type=int, default=80, help="Fixed count for high-sim picks per prototype if high-ratio < 0")
    parser.add_argument("--top-mid", type=int, default=15, help="Fixed count for mid picks per prototype if mid-ratio < 0")
    parser.add_argument("--top-tail", type=int, default=10, help="Fixed count for tail picks per prototype if tail-ratio < 0")
    parser.add_argument("--high-ratio", type=float, default=-1.0, help="If >= 0, high count = ceil(total_candidates * ratio)")
    parser.add_argument("--mid-ratio", type=float, default=-1.0, help="If >= 0, mid count = ceil(len(mid_pool) * ratio)")
    parser.add_argument("--tail-ratio", type=float, default=-1.0, help="If >= 0, tail count = ceil(len(tail_pool) * ratio)")
    parser.add_argument("--mid-width", type=int, default=100, help="Width of the non-overlapping mid pool, starting after high region")
    parser.add_argument("--tail-width", type=int, default=200, help="Width of the non-overlapping tail pool, starting after mid region")
    parser.add_argument("--faiss-topk", type=int, default=-1, help="How many nearest neighbors to request from FAISS. Default: high_k + mid_width + tail_width")
    parser.add_argument("--max-prototypes", type=int, default=-1, help="Optional limit for debugging")
    parser.add_argument("--strict-window-hash", action="store_true", help="If set, fail when window_hash mismatches. Default: skip and warn.")
    return parser.parse_args()
    


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc


def iter_input_records(input_path: Path) -> Iterator[Dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        yield from load_jsonl(input_path)
        return
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(data, dict):
            for key in ("data", "items", "records"):
                if isinstance(data.get(key), list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            yield item
                    return
        raise ValueError("Unsupported JSON structure in input")
    raise ValueError("Input must be .json or .jsonl")


def detect_conversations(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        return conversations
    messages = record.get("messages")
    if isinstance(messages, list):
        return messages
    raise ValueError("Each record must contain 'conversations' or 'messages'.")


def role_of(msg: Dict[str, Any]) -> str:
    return str(msg.get("from") or msg.get("role") or "").strip().lower()


# def value_of(msg: Dict[str, Any]) -> str:
#     value = msg.get("value")
#     if value is None:
#         value = msg.get("content", "")
#     if isinstance(value, list):
#         parts = []
#         for item in value:
#             if isinstance(item, dict):
#                 text = item.get("text") or item.get("content") or ""
#                 if text:
#                     parts.append(str(text))
#             elif item:
#                 parts.append(str(item))
#         return "\n".join(parts)
#     return str(value).strip()
def value_of(msg: Dict[str, Any]) -> str:
    value = msg.get("value")
    if value is None:
        value = msg.get("content")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def canonical_message(msg: Dict[str, Any]) -> Dict[str, str]:
    return {
        "from": role_of(msg),
        "value": value_of(msg),
    }


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def stable_record_id(record: Dict[str, Any]) -> str:
    payload = {
        "conversations": [canonical_message(m) for m in detect_conversations(record)]
    }
    if "tools" in record:
        payload["tools"] = record["tools"]
    return "rec_" + sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def stable_window_hash(window_msgs: Sequence[Dict[str, Any]]) -> str:
    payload = [canonical_message(m) for m in window_msgs]
    return "win_" + sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def load_manifest(emb_dir: Path) -> Dict[str, Any]:
    path = emb_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found in {emb_dir}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prototypes(path: Path, max_prototypes: int = -1) -> List[Dict[str, Any]]:
    items = list(load_jsonl(path))
    out: List[Dict[str, Any]] = []
    for x in items:
        desc = str(x.get("description") or "").strip()
        pid = str(x.get("prototype_id") or "").strip()
        if not desc or not pid:
            continue
        out.append(x)
        if max_prototypes > 0 and len(out) >= max_prototypes:
            break
    if not out:
        raise ValueError(f"No valid prototypes found in {path}")
    return out


def choose_k(total: int, fixed_n: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if ratio >= 0:
        return min(total, max(0, int(math.ceil(total * ratio))))
    return min(total, max(0, fixed_n))


def sample_from_pool(pool_ids: Sequence[int], pool_scores: Sequence[float], k: int, rng: random.Random):
    n = len(pool_ids)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(zip(pool_ids, pool_scores))
    idxs = list(range(n))
    rng.shuffle(idxs)
    idxs = idxs[:k]
    return [(int(pool_ids[i]), float(pool_scores[i])) for i in idxs]


# def load_faiss_index(index_path: Path, nprobe: int):
#     index = faiss.read_index(str(index_path))
#     if hasattr(index, "nprobe"):
#         try:
#             index.nprobe = nprobe
#         except Exception:
#             pass
#     return index
def load_faiss_index(index_path: Path, nprobe: int):
    index = faiss.read_index(str(index_path))
    try:
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", nprobe)
    except Exception:
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
    return index


# def collect_selected_meta_ids(
#     index,
#     proto_embs: np.ndarray,
#     prototypes: List[Dict[str, Any]],
#     total_candidates: int,
#     args: argparse.Namespace,
# ) -> Dict[int, Dict[str, Any]]:
#     """
#     Returns:
#       meta_id -> {
#         'best_score': float,
#         'matches': [{'prototype_id', 'task_type', 'score', 'bucket'}...],
#         'best_match': {...}
#       }
#     """
#     rng = random.Random(args.seed)
#     selected: Dict[int, Dict[str, Any]] = {}

#     for p_idx, proto in enumerate(prototypes):
#         proto_id = str(proto.get("prototype_id"))
#         task_type = str(proto.get("task_type") or "")
#         q = np.asarray(proto_embs[p_idx:p_idx + 1], dtype=np.float32)

#         high_k = choose_k(total_candidates, args.top_high, args.high_ratio)
#         faiss_topk = args.faiss_topk if args.faiss_topk > 0 else high_k + args.mid_width + args.tail_width
#         faiss_topk = min(total_candidates, max(1, faiss_topk))

#         D, I = index.search(q, faiss_topk)
#         ids_all = [int(x) for x in I[0].tolist() if int(x) >= 0]
#         scores_all = [float(D[0][i]) for i, x in enumerate(I[0].tolist()) if int(x) >= 0]
        
#         # NON-overlapping regions:
#         # high = [0 : high_k]
#         # mid  = [high_k : high_k + mid_width]
#         # tail = [high_k + mid_width : high_k + mid_width + tail_width]
#         high_ids = ids_all[:high_k]
#         high_scores = scores_all[:high_k]

#         mid_start = high_k
#         mid_end = min(len(ids_all), mid_start + args.mid_width)
#         mid_ids = ids_all[mid_start:mid_end]
#         mid_scores = scores_all[mid_start:mid_end]
#         mid_k = choose_k(len(mid_ids), args.top_mid, args.mid_ratio)
#         mid_pick = sample_from_pool(mid_ids, mid_scores, mid_k, rng)

#         tail_start = mid_end
#         tail_end = min(len(ids_all), tail_start + args.tail_width)
#         tail_ids = ids_all[tail_start:tail_end]
#         tail_scores = scores_all[tail_start:tail_end]
#         tail_k = choose_k(len(tail_ids), args.top_tail, args.tail_ratio)
#         tail_pick = sample_from_pool(tail_ids, tail_scores, tail_k, rng)

#         picks = []
#         picks.extend([(int(i), float(s), "high") for i, s in zip(high_ids, high_scores)])
#         picks.extend([(int(i), float(s), "mid") for i, s in mid_pick])
#         picks.extend([(int(i), float(s), "tail") for i, s in tail_pick])

#         for meta_id, score, bucket in picks:
#             match = {
#                 "prototype_id": proto_id,
#                 "task_type": task_type,
#                 "score": score,
#                 "bucket": bucket,
#             }
#             if meta_id not in selected:
#                 selected[meta_id] = {
#                     "best_score": score,
#                     "best_match": match,
#                     "matches": [match],
#                 }
#             else:
#                 selected[meta_id]["matches"].append(match)
#                 if score > selected[meta_id]["best_score"]:
#                     selected[meta_id]["best_score"] = score
#                     selected[meta_id]["best_match"] = match

#         print(
#             f"[INFO] prototype={proto_id} task={task_type} "
#             f"high={len(high_ids)} mid={len(mid_pick)} tail={len(tail_pick)} "
#             f"faiss_topk={faiss_topk}",
#             file=sys.stderr,
#         )

#     return selected
def collect_selected_meta_ids(index, proto_embs: np.ndarray, prototypes: List[Dict[str, Any]], total_candidates: int, args: argparse.Namespace):
    rng = random.Random(args.seed)
    selected: Dict[int, Dict[str, Any]] = {}

    for p_idx, proto in enumerate(prototypes):
        proto_id = str(proto.get("prototype_id"))
        task_type = str(proto.get("task_type") or "")
        q = np.asarray(proto_embs[p_idx:p_idx + 1], dtype=np.float32)

        high_k_requested = choose_k(total_candidates, args.top_high, args.high_ratio)

        faiss_topk = args.faiss_topk if args.faiss_topk > 0 else high_k_requested + args.mid_width + args.tail_width
        faiss_topk = min(total_candidates, max(1, faiss_topk))

        D, I = index.search(q, faiss_topk)
        ids_all = [int(x) for x in I[0].tolist() if int(x) >= 0]
        scores_all = [float(D[0][i]) for i, x in enumerate(I[0].tolist()) if int(x) >= 0]

        # 关键：用实际可用长度裁一下
        high_k = min(high_k_requested, len(ids_all))

        high_ids = ids_all[:high_k]
        high_scores = scores_all[:high_k]

        mid_start = high_k
        mid_end = min(len(ids_all), mid_start + args.mid_width)
        mid_ids = ids_all[mid_start:mid_end]
        mid_scores = scores_all[mid_start:mid_end]

        mid_k = choose_k(len(mid_ids), args.top_mid, args.mid_ratio)
        mid_pick = sample_from_pool(mid_ids, mid_scores, mid_k, rng)

        tail_start = mid_end
        tail_end = min(len(ids_all), tail_start + args.tail_width)
        tail_ids = ids_all[tail_start:tail_end]
        tail_scores = scores_all[tail_start:tail_end]

        tail_k = choose_k(len(tail_ids), args.top_tail, args.tail_ratio)
        tail_pick = sample_from_pool(tail_ids, tail_scores, tail_k, rng)

        print(
            f"[INFO] prototype={proto_id} task={task_type} "
            f"high={len(high_ids)} mid={len(mid_pick)} tail={len(tail_pick)} "
            f"faiss_topk={faiss_topk} actual_n={len(ids_all)}"
        )

        picks = []
        picks.extend([(int(i), float(s), "high") for i, s in zip(high_ids, high_scores)])
        picks.extend([(int(i), float(s), "mid") for i, s in mid_pick])
        picks.extend([(int(i), float(s), "tail") for i, s in tail_pick])

        for meta_id, score, bucket in picks:
            match = {
                "prototype_id": proto_id,
                "task_type": task_type,
                "score": score,
                "bucket": bucket,
            }
            if meta_id not in selected:
                selected[meta_id] = {
                    "best_score": score,
                    "best_match": match,
                    "matches": [match],
                }
            else:
                selected[meta_id]["matches"].append(match)
                if score > selected[meta_id]["best_score"]:
                    selected[meta_id]["best_score"] = score
                    selected[meta_id]["best_match"] = match

    return selected

def load_selected_metadata(metadata_path: Path, needed_meta_ids: set[int]) -> Dict[int, Dict[str, Any]]:
    found: Dict[int, Dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx not in needed_meta_ids:
                continue
            line = line.strip()
            if not line:
                continue
            found[line_idx] = json.loads(line)
            if len(found) >= len(needed_meta_ids):
                break
    missing = needed_meta_ids - set(found.keys())
    if missing:
        raise ValueError(f"Failed to find {len(missing)} metadata rows from metadata.jsonl")
    return found


def group_targets_by_record(selected_meta: Dict[int, Dict[str, Any]], selected_info: Dict[int, Dict[str, Any]]):
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for meta_id, meta in selected_meta.items():
        record_id = str(meta.get("record_id") or "")
        if not record_id:
            raise ValueError("metadata row missing record_id. Rebuild metadata first.")
        out[record_id].append({
            "meta_id": meta_id,
            "uid": meta["uid"],
            "record_id": record_id,
            "conversation_id": meta.get("conversation_id", ""),
            "gpt_turn_index": int(meta["gpt_turn_index"]),
            "start_idx": int(meta["start_idx"]),
            "end_idx": int(meta["end_idx"]),
            "window_hash": meta["window_hash"],
            "retrieval_text": meta.get("retrieval_text", ""),
            "has_tools": bool(meta.get("has_tools", False)),
            "source": meta.get("source", "unknown"),
            "best_score": float(selected_info[meta_id]["best_score"]),
            "best_match": selected_info[meta_id]["best_match"],
            "matches": selected_info[meta_id]["matches"],
        })
    for k in out:
        out[k].sort(key=lambda x: (x["end_idx"], x["start_idx"], x["uid"]))
    return out


def build_training_record(record: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    msgs = detect_conversations(record)
    sliced = msgs[target["start_idx"]: target["end_idx"] + 1]
    return {
        "id": target["uid"],
        "record_id": target["record_id"],
        "parent_conversation_id": target["conversation_id"],
        "source": str(record.get("source") or record.get("dataset") or target.get("source") or "unknown"),
        "prototype_id": target["best_match"]["prototype_id"],
        "prototype_task_type": target["best_match"]["task_type"],
        "prototype_bucket": target["best_match"]["bucket"],
        "prototype_score": target["best_match"]["score"],
        "conversations": sliced,
        **({"tools": record["tools"]} if "tools" in record else {}),
    }


def build_debug_record(target: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "uid": target["uid"],
        "record_id": target["record_id"],
        "conversation_id": target["conversation_id"],
        "gpt_turn_index": target["gpt_turn_index"],
        "start_idx": target["start_idx"],
        "end_idx": target["end_idx"],
        "window_hash": target["window_hash"],
        "has_tools": target["has_tools"],
        "source": target["source"],
        "best_score": target["best_score"],
        "best_match": target["best_match"],
        "matches": target["matches"],
        "retrieval_text": target["retrieval_text"],
    }


def reconstruct_selected_records(input_path: Path, targets_by_record: Dict[str, List[Dict[str, Any]]], output_path: Path, debug_path: Optional[Path], warnings_path: Optional[Path], strict_window_hash: bool):
    written = 0
    found_targets = 0
    total_targets = sum(len(v) for v in targets_by_record.values())

    with output_path.open("w", encoding="utf-8") as out_f:
        debug_f = debug_path.open("w", encoding="utf-8") if debug_path else None
        warn_f = warnings_path.open("w", encoding="utf-8") if warnings_path else None
        try:
            for _, record in enumerate(iter_input_records(input_path)):
                rid = stable_record_id(record)
                targets = targets_by_record.get(rid)
                if not targets:
                    continue

                msgs = detect_conversations(record)
                for target in targets:
                    sliced = msgs[target["start_idx"]: target["end_idx"] + 1]
                    cur_hash = stable_window_hash(sliced)
                    if cur_hash != target["window_hash"]:
                        warn = {
                            "type": "window_hash_mismatch",
                            "uid": target["uid"],
                            "record_id": rid,
                            "expected_window_hash": target["window_hash"],
                            "actual_window_hash": cur_hash,
                            "start_idx": target["start_idx"],
                            "end_idx": target["end_idx"],
                        }
                        if warn_f is not None:
                            warn_f.write(json.dumps(warn, ensure_ascii=False) + "\n")
                        if strict_window_hash:
                            raise ValueError(f"window_hash mismatch for {target['uid']}")
                        continue

                    out_f.write(json.dumps(build_training_record(record, target), ensure_ascii=False) + "\n")
                    written += 1
                    found_targets += 1
                    if debug_f is not None:
                        debug_f.write(json.dumps(build_debug_record(target), ensure_ascii=False) + "\n")

                if found_targets >= total_targets:
                    break
        finally:
            if debug_f is not None:
                debug_f.close()
            if warn_f is not None:
                warn_f.close()
    return written, found_targets, total_targets


def main():
    args = parse_args()
    input_path = Path(args.input)
    prototypes_path = Path(args.prototypes)
    emb_dir = Path(args.emb_dir)
    index_path = Path(args.index)
    output_path = Path(args.output)
    debug_path = Path(args.debug_output) if args.debug_output else None
    warnings_path = Path(args.warnings_output) if args.warnings_output else None

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if debug_path and debug_path.exists() and not args.overwrite:
        raise FileExistsError(f"Debug output already exists: {debug_path}")

    manifest = load_manifest(emb_dir)
    total_candidates = int(manifest["total_records"])
    model_name = args.model or str(manifest.get("model") or "BAAI/bge-m3")
    normalize_queries = bool(args.normalize_queries or manifest.get("normalize_embeddings", False))

    metadata_path = emb_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {emb_dir}")

    prototypes = load_prototypes(prototypes_path, max_prototypes=args.max_prototypes)

    model = SentenceTransformer(
        model_name,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": f"{args.device}"},
        tokenizer_kwargs={"padding_side": "left"},
        local_files_only=True
    )
    proto_embs = model.encode(
        [str(x["description"]) for x in prototypes],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_queries,
        show_progress_bar=True,
        prompt="Given a prototype query, retrieve relevant conversations that match the prototype"
    ).astype(np.float32)

    print(f"[INFO] Loading FAISS index: {index_path}", file=sys.stderr)
    index = load_faiss_index(index_path, args.nprobe)
    print(f"[INFO] Index ntotal={index.ntotal}", file=sys.stderr)

    selected_info = collect_selected_meta_ids(
        index=index,
        proto_embs=proto_embs,
        prototypes=prototypes,
        total_candidates=total_candidates,
        args=args,
    )
    needed_meta_ids = set(selected_info.keys())
    print(f"[INFO] Unique selected meta rows: {len(needed_meta_ids)}", file=sys.stderr)
    
    print("[INFO] Loading selected metadata rows...", file=sys.stderr)
    selected_meta = load_selected_metadata(metadata_path, needed_meta_ids)
    targets_by_record = group_targets_by_record(selected_meta, selected_info)
    print(f"[INFO] Conversations with selected windows: {len(targets_by_record)}", file=sys.stderr)

    print("[INFO] Reconstructing ShareGPT sub-conversations...", file=sys.stderr)

    written, found, total = reconstruct_selected_records(
        input_path=input_path,
        targets_by_record=targets_by_record,
        output_path=output_path,
        debug_path=debug_path,
        warnings_path=warnings_path,
        strict_window_hash=args.strict_window_hash,
    )
    if found < len(needed_meta_ids):
        print(
            f"[WARN] Only reconstructed {found}/{len(needed_meta_ids)} selected windows. "
            f"Check whether --input matches the source used to build embeddings.",
            file=sys.stderr,
        )

    summary = {
        "prototypes": len(prototypes),
        "total_candidates": total_candidates,
        "selected_unique_meta_rows": len(needed_meta_ids),
        "skipped_due_to_mismatch_or_missing": total - written,
        "targets_total": total,
        "reconstructed_windows": written,
        "output": str(output_path),
        "debug_output": str(debug_path) if debug_path else "",
        "warnings_output": str(warnings_path) if warnings_path else "",
        "selection": {
            "top_high": args.top_high,
            "top_mid": args.top_mid,
            "top_tail": args.top_tail,
            "high_ratio": args.high_ratio,
            "mid_ratio": args.mid_ratio,
            "tail_ratio": args.tail_ratio,
            "mid_width": args.mid_width,
            "tail_width": args.tail_width,
            "faiss_topk": args.faiss_topk,
            "nprobe": args.nprobe,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)



if __name__ == "__main__":
    main()
