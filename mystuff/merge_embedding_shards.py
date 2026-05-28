#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple embedding shard directories produced by build_candidate_embeddings_v2.py
into a single emb-cache directory compatible with build_faiss_index.py.

Each shard dir should contain:
- embeddings_*.npy
- metadata.jsonl
- manifest.json

Typical usage:
python merge_embedding_shards.py \
  --input-root /path/to/emb_shards \
  --output-dir /path/to/emb_cache_merged

Notes:
- Chunk files will be renumbered globally as embeddings_000000.npy, embeddings_000001.npy, ...
- metadata.jsonl rows are concatenated in the same order as chunks are copied.
- manifest.json is regenerated for the merged cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Merge embedding shard directories.")
    p.add_argument("--input-root", required=True, help="Root directory containing shard subdirs")
    p.add_argument("--output-dir", required=True, help="Merged output directory")
    p.add_argument("--pattern", default="emb_*", help="Glob for shard subdirectories")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output dir")
    return p.parse_args()


def ensure_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty. Use --overwrite to continue.")
        for p in path.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir, args.overwrite)

    shard_dirs = sorted([p for p in input_root.glob(args.pattern) if p.is_dir()])
    if not shard_dirs:
        raise FileNotFoundError(f"No shard dirs matched {args.pattern} under {input_root}")

    merged_metadata = output_dir / "metadata.jsonl"
    chunk_idx = 0
    total_records = 0
    total_chunks = 0

    merged_model = None
    merged_dim = None
    merged_dtype = None
    merged_norm = None
    merged_chunk_size = None
    input_paths: List[str] = []

    with merged_metadata.open("w", encoding="utf-8") as meta_out:
        for shard_dir in shard_dirs:
            manifest_path = shard_dir / "manifest.json"
            metadata_path = shard_dir / "metadata.jsonl"
            if not manifest_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Missing manifest.json or metadata.jsonl in {shard_dir}")

            manifest = load_manifest(manifest_path)
            input_paths.append(str(manifest.get("input", "")))

            if merged_model is None:
                merged_model = manifest.get("model")
                merged_dim = manifest.get("embedding_dim")
                merged_dtype = manifest.get("dtype")
                merged_norm = manifest.get("normalize_embeddings")
                merged_chunk_size = manifest.get("chunk_size")
            else:
                if manifest.get("embedding_dim") != merged_dim:
                    raise ValueError(f"Embedding dim mismatch in {shard_dir}")
                if manifest.get("dtype") != merged_dtype:
                    raise ValueError(f"Dtype mismatch in {shard_dir}")
                if manifest.get("normalize_embeddings") != merged_norm:
                    raise ValueError(f"normalize_embeddings mismatch in {shard_dir}")

            chunk_files = sorted(shard_dir.glob("embeddings_*.npy"))
            if not chunk_files:
                raise FileNotFoundError(f"No embeddings_*.npy files in {shard_dir}")

            shard_rows = 0
            for chunk_path in chunk_files:
                arr = np.load(chunk_path, mmap_mode="r")
                out_chunk = output_dir / f"embeddings_{chunk_idx:06d}.npy"
                np.save(out_chunk, np.asarray(arr))
                chunk_idx += 1
                total_chunks += 1
                shard_rows += int(arr.shape[0])

            with metadata_path.open("r", encoding="utf-8") as meta_in:
                for line in meta_in:
                    if not line.strip():
                        continue
                    meta_out.write(line)
                    total_records += 1

            print(f"[INFO] merged {shard_dir.name}: rows={shard_rows}, chunks={len(chunk_files)}")

    merged_manifest = {
        "model": merged_model,
        "device": "merged",
        "embedding_dim": merged_dim,
        "dtype": merged_dtype,
        "normalize_embeddings": merged_norm,
        "chunk_size": merged_chunk_size,
        "total_records": total_records,
        "total_chunks": total_chunks,
        "input": input_paths,
        "merged_from": [str(p) for p in shard_dirs],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(merged_manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] total_records={total_records}, total_chunks={total_chunks}")


if __name__ == "__main__":
    main()
