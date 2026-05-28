#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a FAISS index from chunked embedding files produced by build_candidate_embeddings.py.

Expected input dir contents:
- embeddings_000000.npy
- embeddings_000001.npy
- ...
- metadata.jsonl
- manifest.json

Index IDs are assigned sequentially in the same order as metadata.jsonl rows.
So returned FAISS ids can be mapped back to metadata by line number / offset.

Typical usage:
python build_faiss_index.py \
  --emb-dir /path/to/emb_cache \
  --output /path/to/emb_cache/faiss_ivf_flat.index \
  --index-type ivf_flat \
  --nlist 4096 \
  --batch-size 100000

For cosine similarity, embeddings should already be normalized and this script uses
METRIC_INNER_PRODUCT by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "faiss is not installed. Please install faiss-cpu or faiss-gpu first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from chunked embedding files.")
    parser.add_argument("--emb-dir", required=True, help="Directory containing embeddings_*.npy and metadata.jsonl.")
    parser.add_argument("--output", required=True, help="Output FAISS index path.")
    parser.add_argument(
        "--index-type",
        default="ivf_flat",
        choices=["flat", "ivf_flat", "ivf_pq"],
        help="FAISS index type.",
    )
    parser.add_argument(
        "--metric",
        default="ip",
        choices=["ip", "l2"],
        help="Distance metric. Use 'ip' if embeddings are normalized and you want cosine search.",
    )
    parser.add_argument("--batch-size", type=int, default=100000, help="How many vectors to add per batch.")
    parser.add_argument("--train-size", type=int, default=200000, help="How many vectors to sample for IVF training.")
    parser.add_argument("--nlist", type=int, default=4096, help="Number of IVF clusters.")
    parser.add_argument("--m", type=int, default=32, help="PQ subquantizers for ivf_pq.")
    parser.add_argument("--nbits", type=int, default=8, help="Bits per PQ code for ivf_pq.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU during training/add if faiss-gpu is available.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for faiss-gpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def load_manifest(emb_dir: Path) -> dict:
    manifest_path = emb_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {emb_dir}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_embedding_files(emb_dir: Path) -> List[Path]:
    files = sorted(emb_dir.glob("embeddings_*.npy"))
    if not files:
        raise FileNotFoundError(f"No embeddings_*.npy files found in {emb_dir}")
    return files


def infer_metric(metric_name: str) -> int:
    if metric_name == "ip":
        return faiss.METRIC_INNER_PRODUCT
    return faiss.METRIC_L2


def make_base_index(dim: int, args: argparse.Namespace):
    metric = infer_metric(args.metric)
    if args.index_type == "flat":
        if metric == faiss.METRIC_INNER_PRODUCT:
            return faiss.IndexFlatIP(dim)
        return faiss.IndexFlatL2(dim)

    if args.index_type == "ivf_flat":
        quantizer = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
        return faiss.IndexIVFFlat(quantizer, dim, args.nlist, metric)

    if args.index_type == "ivf_pq":
        quantizer = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
        return faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, args.nbits, metric)

    raise ValueError(f"Unsupported index type: {args.index_type}")


def maybe_to_gpu(index, args: argparse.Namespace):
    if not args.use_gpu:
        return index, None
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, args.gpu_id, index)
        return gpu_index, res
    except Exception as exc:
        print(f"[WARN] Failed to move FAISS index to GPU: {exc}. Falling back to CPU.", file=sys.stderr)
        return index, None


def maybe_to_cpu(index):
    try:
        return faiss.index_gpu_to_cpu(index)
    except Exception:
        return index


def iter_chunks(files: List[Path]) -> Iterator[Tuple[Path, np.ndarray]]:
    for path in files:
        arr = np.load(path, mmap_mode="r")
        yield path, arr


def sample_training_vectors(files: List[Path], train_size: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples: List[np.ndarray] = []
    collected = 0

    counts: List[Tuple[Path, int]] = []
    total = 0
    for p in files:
        arr = np.load(p, mmap_mode="r")
        n = int(arr.shape[0])
        counts.append((p, n))
        total += n

    if total == 0:
        raise ValueError("No vectors found.")

    target = min(train_size, total)

    for p, n in counts:
        if collected >= target:
            break
        # proportional sample from each chunk, with a floor of 1 if possible
        share = max(1, int(round(target * (n / total))))
        share = min(share, n, target - collected)
        if share <= 0:
            continue
        arr = np.load(p, mmap_mode="r")
        idx = rng.choice(n, size=share, replace=False)
        block = np.asarray(arr[idx], dtype=np.float32)
        samples.append(block)
        collected += block.shape[0]

    out = np.vstack(samples)
    if out.shape[0] > target:
        out = out[:target]
    if out.shape[1] != dim:
        raise ValueError(f"Training vectors dim mismatch: got {out.shape[1]}, expected {dim}")
    return out


def add_vectors_with_ids(index, files: List[Path], batch_size: int, start_id: int = 0) -> int:
    next_id = start_id
    total_added = 0
    for file_idx, p in enumerate(files):
        arr = np.load(p, mmap_mode="r")
        n = int(arr.shape[0])
        print(f"[INFO] Adding chunk {file_idx + 1}/{len(files)}: {p.name}, rows={n}", file=sys.stderr)
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            block = np.asarray(arr[s:e], dtype=np.float32)
            ids = np.arange(next_id, next_id + (e - s), dtype=np.int64)
            index.add_with_ids(block, ids)
            next_id += (e - s)
            total_added += (e - s)
            if total_added % 500000 == 0:
                print(f"[INFO] Added {total_added} vectors so far", file=sys.stderr)
    return total_added


def count_metadata_rows(metadata_path: Path) -> int:
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")
    count = 0
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_sidecar(output_path: Path, *, args: argparse.Namespace, dim: int, total_vectors: int, metadata_rows: int) -> None:
    sidecar = {
        "index_path": str(output_path),
        "index_type": args.index_type,
        "metric": args.metric,
        "dim": dim,
        "total_vectors": total_vectors,
        "metadata_rows": metadata_rows,
        "id_to_metadata": "faiss_id corresponds to metadata.jsonl 0-based row index",
        "nlist": args.nlist if args.index_type in {"ivf_flat", "ivf_pq"} else None,
        "m": args.m if args.index_type == "ivf_pq" else None,
        "nbits": args.nbits if args.index_type == "ivf_pq" else None,
    }
    with output_path.with_suffix(output_path.suffix + ".meta.json").open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    emb_dir = Path(args.emb_dir)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(emb_dir)
    files = list_embedding_files(emb_dir)
    metadata_rows = count_metadata_rows(emb_dir / "metadata.jsonl")

    dim = int(manifest.get("embedding_dim") or 0)
    if dim <= 0:
        # fallback from first chunk
        first = np.load(files[0], mmap_mode="r")
        dim = int(first.shape[1])
    print(f"[INFO] embedding_dim={dim}", file=sys.stderr)
    print(f"[INFO] chunks={len(files)}", file=sys.stderr)
    print(f"[INFO] metadata_rows={metadata_rows}", file=sys.stderr)

    base_index = make_base_index(dim, args)
    index = faiss.IndexIDMap2(base_index)
    index, _gpu_res = maybe_to_gpu(index, args)

    if args.index_type in {"ivf_flat", "ivf_pq"}:
        print(f"[INFO] Sampling training vectors: train_size={args.train_size}", file=sys.stderr)
        train_vecs = sample_training_vectors(files, args.train_size, dim, args.seed)
        print(f"[INFO] Training index on {train_vecs.shape[0]} vectors", file=sys.stderr)
        index.train(train_vecs)
        print("[INFO] Index training finished", file=sys.stderr)

    total_vectors = add_vectors_with_ids(index, files, args.batch_size, start_id=0)
    print(f"[INFO] Total added vectors: {total_vectors}", file=sys.stderr)

    cpu_index = maybe_to_cpu(index)
    faiss.write_index(cpu_index, str(output_path))
    print(f"[DONE] Saved index to: {output_path}", file=sys.stderr)

    write_sidecar(
        output_path,
        args=args,
        dim=dim,
        total_vectors=total_vectors,
        metadata_rows=metadata_rows,
    )
    print(f"[DONE] Saved sidecar to: {output_path}.meta.json", file=sys.stderr)

    if metadata_rows != total_vectors:
        print(
            f"[WARN] metadata rows ({metadata_rows}) != vectors in index ({total_vectors}). Please verify input consistency.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
