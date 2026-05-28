#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_candidate_embeddings_v3.py

Compared with v2:
- stable record_id based on conversation content (+ tools)
- stable uid based on record_id + window offsets
- window_hash stored in metadata for later verification
- still writes chunked embeddings + metadata.jsonl + manifest.json

This is the "correct future version" for rebuilding embeddings.
If you do NOT want to rebuild embeddings now, use rebuild_metadata_only_v1.py
and retrieve_by_prototype_faiss_v2.py instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "assistant"}
TOOL_ROLES = {"function_call", "tool", "function", "observation"}
SYSTEM_ROLES = {"system"}


@dataclass
class WindowRecord:
    uid: str
    record_id: str
    conversation_id: str
    gpt_turn_index: int
    start_idx: int
    end_idx: int
    window_hash: str
    retrieval_text: str
    has_tools: bool
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and save embeddings for ShareGPT candidate windows.")
    parser.add_argument("--input", required=True, help="Input ShareGPT .json or .jsonl file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save embeddings and metadata.")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B", help="SentenceTransformer model name or path.")
    parser.add_argument("--device", default="cuda", help="Encoding device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="How many windows to save per chunk.")
    parser.add_argument("--max-context-turns", type=int, default=6, help="Max previous turns kept in retrieval text/window context.")
    parser.add_argument("--assistant-prefix-chars", type=int, default=160, help="Keep at most this many chars from final assistant reply in retrieval_text.")
    parser.add_argument("--max-message-chars", type=int, default=256, help="Cap every message to this many chars before concatenation.")
    parser.add_argument("--max-total-chars", type=int, default=1200, help="Cap the final retrieval_text length.")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Force model.max_seq_length to this value. Set <=0 to keep model default.")
    parser.add_argument("--normalize-embeddings", action="store_true", help="Normalize embeddings before saving.")
    parser.add_argument("--fp16", action="store_true", help="Save embeddings as float16 instead of float32.")
    parser.add_argument("--inference-fp16", action="store_true", help="Run model inference in fp16 on CUDA.")
    parser.add_argument("--empty-cache-every", type=int, default=50, help="Call torch.cuda.empty_cache() every N encode steps on CUDA. 0 disables.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dir contents.")
    parser.add_argument("--limit-conversations", type=int, default=-1, help="Optional limit for debugging.")
    parser.add_argument("--limit-windows", type=int, default=-1, help="Optional limit for debugging.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing metadata/chunks if present.")
    return parser.parse_args()



def ensure_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite and resume:
            raise ValueError("--overwrite and --resume cannot be used together.")
        if overwrite:
            for p in output_dir.iterdir():
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    import shutil
                    shutil.rmtree(p)
        elif not resume:
            raise FileExistsError(f"Output dir {output_dir} is not empty. Use --overwrite or --resume.")
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_input_records(input_path: Path):
    suffixes = [s.lower() for s in input_path.suffixes]
    is_gzip = suffixes and suffixes[-1] == ".gz"

    open_func = gzip.open if is_gzip else open

    def decode_line(raw: bytes) -> str:
        # 先尝试常见编码
        for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        # 最后兜底：替换非法字节，尽量保住 JSON 结构
        return raw.decode("utf-8", errors="replace")

    # 只推荐处理 jsonl；大文件 json 不建议这样搞
    if ".jsonl" in suffixes or input_path.suffix.lower() == ".jsonl":
        with open_func(input_path, "rb") as f:
            for line_no, raw_line in enumerate(f, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                line = decode_line(raw_line).strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] skip bad json line {line_no}", file=sys.stderr)
                    continue
        return

    # 如果是 .json，小文件才建议这样读
    if input_path.suffix.lower() == ".json":
        raw = input_path.read_bytes()

        text = None
        for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            text = raw.decode("utf-8", errors="replace")

        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        elif isinstance(data, dict):
            for key in ("data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            yield item
                    return
            raise ValueError("Unsupported JSON structure.")
        else:
            raise ValueError("Unsupported JSON structure.")
        return

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
    return str(value).strip()


def short_role(role: str) -> str:
    # if role in USER_ROLES:
    #     return "用户"
    # if role in ASSISTANT_ROLES:
    #     return "助手"
    # if role == "function_call":
    #     return "工具调用"
    # if role in {"observation", "tool", "function"}:
    #     return "工具结果"
    # if role in SYSTEM_ROLES:
    #     return "系统"
    return role


def clip_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    return s[:max_chars]


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


def build_retrieval_text(
    window_msgs: Sequence[Dict[str, Any]],
    assistant_prefix_chars: int,
    max_message_chars: int,
    max_total_chars: int,
) -> str:
    lines: List[str] = []
    for idx, msg in enumerate(window_msgs):
        role = role_of(msg)
        value = value_of(msg)
        if not value:
            continue
        if idx == len(window_msgs) - 1 and role in ASSISTANT_ROLES:
            value = clip_text(value, assistant_prefix_chars)
        else:
            value = clip_text(value, max_message_chars)
        lines.append(f"{short_role(role)}：{value}")
    text = "\n".join(lines).strip()
    if max_total_chars > 0 and len(text) > max_total_chars:
        # Keep the most recent part, which is usually more useful for retrieval.
        text = text[-max_total_chars:]
    return text


def slice_windows(
    record: Dict[str, Any],
    conv_idx: int,
    max_context_turns: int,
    assistant_prefix_chars: int,
    max_message_chars: int,
    max_total_chars: int,
) -> Iterator[Tuple[WindowRecord, List[Dict[str, Any]]]]:
    msgs = detect_conversations(record)
    display_conversation_id = str(record.get("id") or record.get("conversation_id") or "")
    source = str(record.get("source") or record.get("dataset") or "unknown")
    record_id = stable_record_id(record)

    for i, msg in enumerate(msgs):
        if role_of(msg) not in ASSISTANT_ROLES:
            continue
        start_idx = max(0, i - max_context_turns)
        window_msgs = msgs[start_idx:i + 1]
        retrieval_text = build_retrieval_text(window_msgs, assistant_prefix_chars, max_message_chars, max_total_chars)
        if not retrieval_text:
            continue
        uid = f"{record_id}__gpt_{i}__{start_idx}_{i}"
        yield WindowRecord(
            uid=uid,
            record_id=record_id,
            conversation_id=display_conversation_id,
            gpt_turn_index=i,
            start_idx=start_idx,
            end_idx=i,
            window_hash=stable_window_hash(window_msgs),
            retrieval_text=retrieval_text,
            has_tools=any(role_of(x) in TOOL_ROLES for x in window_msgs),
            source=source,
        ), list(window_msgs)


def count_existing_metadata(metadata_path: Path) -> int:
    if not metadata_path.exists():
        return 0
    c = 0
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


def next_chunk_index(output_dir: Path) -> int:
    max_idx = -1
    for p in output_dir.glob("embeddings_*.npy"):
        try:
            idx = int(p.stem.split("_")[-1])
            max_idx = max(max_idx, idx)
        except Exception:
            pass
    return max_idx + 1


def save_chunk(output_dir: Path, chunk_index: int, embeddings: np.ndarray, dtype: np.dtype) -> Path:
    out_path = output_dir / f"embeddings_{chunk_index:06d}.npy"
    np.save(out_path, embeddings.astype(dtype, copy=False))
    return out_path


def append_metadata(metadata_path: Path, records: Sequence[WindowRecord]) -> None:
    with metadata_path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def write_manifest(
    output_dir: Path,
    *,
    model_name: str,
    device: str,
    embedding_dim: int,
    dtype: str,
    normalize_embeddings: bool,
    chunk_size: int,
    total_records: int,
    total_chunks: int,
    input_path: str,
    max_seq_length: int,
) -> None:
    manifest = {
        "model": model_name,
        "device": device,
        "embedding_dim": embedding_dim,
        "dtype": dtype,
        "normalize_embeddings": normalize_embeddings,
        "chunk_size": chunk_size,
        "total_records": total_records,
        "total_chunks": total_chunks,
        "input": input_path,
        "max_seq_length": max_seq_length,
        "id_scheme": "stable_record_id + window_hash",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def flush_chunk(output_dir: Path, chunk_index: int, emb_buffer: List[np.ndarray], meta_buffer: List[WindowRecord], dtype: np.dtype):
    if not emb_buffer:
        return chunk_index, 0
    arr = np.vstack(emb_buffer)
    save_chunk(output_dir, chunk_index, arr, dtype)
    append_metadata(output_dir / "metadata.jsonl", meta_buffer)
    written = arr.shape[0]
    emb_buffer.clear()
    meta_buffer.clear()
    return chunk_index + 1, written


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)

    metadata_path = output_dir / "metadata.jsonl"
    skip_records = count_existing_metadata(metadata_path) if args.resume else 0
    chunk_index = next_chunk_index(output_dir) if args.resume else 0

    model = SentenceTransformer(
        args.model,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": f"{args.device}"},
        tokenizer_kwargs={"padding_side": "left"},
        local_files_only=True,
    )
    if args.max_seq_length > 0:
        model.max_seq_length = int(args.max_seq_length)
    if args.inference_fp16 and str(args.device).startswith("cuda"):
        try:
            model.half()
        except Exception:
            pass

    emb_buffer: List[np.ndarray] = []
    meta_buffer: List[WindowRecord] = []
    batch_texts: List[str] = []
    batch_meta: List[WindowRecord] = []
    dtype = np.float16 if args.fp16 else np.float32

    total_windows_seen = 0
    total_windows_saved = 0
    total_conversations = 0
    embedding_dim: Optional[int] = None
    encode_steps = 0

    def encode_and_buffer():
        nonlocal batch_texts, batch_meta, emb_buffer, meta_buffer, total_windows_saved, chunk_index, embedding_dim, encode_steps
        if not batch_texts:
            return
        # embs = model.encode(
        #     batch_texts,
        #     batch_size=args.batch_size,
        #     convert_to_numpy=True,
        #     normalize_embeddings=args.normalize_embeddings,
        #     show_progress_bar=False,
        # )
        try:
            embs = model.encode(
                batch_texts,
                batch_size=args.batch_size,
                normalize_embeddings=args.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            # raise
            embs = model.encode(
                batch_texts,
                batch_size=int(args.batch_size // 2),
                normalize_embeddings=args.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        encode_steps += 1
        if args.empty_cache_every > 0 and str(args.device).startswith("cuda") and encode_steps % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        if embedding_dim is None:
            embedding_dim = int(embs.shape[1])
        for vec, rec in zip(embs, batch_meta):
            emb_buffer.append(vec[np.newaxis, :])
            meta_buffer.append(rec)
            if len(meta_buffer) >= args.chunk_size:
                chunk_index, written = flush_chunk(output_dir, chunk_index, emb_buffer, meta_buffer, dtype)
                total_windows_saved += written
        batch_texts = []
        batch_meta = []

    for conv_idx, record in enumerate(iter_input_records(input_path)):
        total_conversations += 1
        if 0 <= args.limit_conversations < total_conversations:
            break
        for rec, _ in slice_windows(
            record,
            conv_idx=conv_idx,
            max_context_turns=args.max_context_turns,
            assistant_prefix_chars=args.assistant_prefix_chars,
            max_message_chars=args.max_message_chars,
            max_total_chars=args.max_total_chars,
        ):
            total_windows_seen += 1
            if args.resume and total_windows_seen <= skip_records:
                continue
            batch_texts.append(rec.retrieval_text)
            batch_meta.append(rec)
            if 0 < args.limit_windows <= (total_windows_seen - skip_records):
                break
            if len(batch_texts) >= args.batch_size:
                encode_and_buffer()
        if 0 < args.limit_windows <= (total_windows_seen - skip_records):
            break

    encode_and_buffer()
    chunk_index, written = flush_chunk(output_dir, chunk_index, emb_buffer, meta_buffer, dtype)
    total_windows_saved += written
    write_manifest(
        output_dir,
        model_name=args.model,
        device=args.device,
        embedding_dim=embedding_dim or 0,
        dtype=str(np.dtype(dtype)),
        normalize_embeddings=bool(args.normalize_embeddings),
        chunk_size=args.chunk_size,
        total_records=skip_records + total_windows_saved,
        total_chunks=len(list(output_dir.glob("embeddings_*.npy"))),
        input_path=str(input_path),
        max_seq_length=int(args.max_seq_length),
    )
    print(json.dumps({
        "windows_seen": total_windows_seen,
        "windows_saved_this_run": total_windows_saved,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
