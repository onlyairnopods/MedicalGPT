#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a large JSONL file into multiple shards for parallel embedding jobs.

Typical usage:
python split_jsonl_shards.py \
  --input /path/to/sharegpt.jsonl \
  --output-dir /path/to/shards \
  --num-shards 8

Or split by lines per shard:
python split_jsonl_shards.py \
  --input /path/to/sharegpt.jsonl \
  --output-dir /path/to/shards \
  --lines-per-shard 250000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split a large JSONL file into shards.")
    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--output-dir", required=True, help="Directory to write shard files")
    p.add_argument("--num-shards", type=int, default=0, help="Number of output shards")
    p.add_argument("--lines-per-shard", type=int, default=0, help="Lines per shard")
    p.add_argument("--prefix", default="sharegpt_shard", help="Output shard prefix")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output dir contents")
    return p.parse_args()


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def ensure_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty. Use --overwrite to continue.")
        for p in path.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                import shutil
                shutil.rmtree(p)
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir, args.overwrite)

    if (args.num_shards > 0) == (args.lines_per_shard > 0):
        raise ValueError("Specify exactly one of --num-shards or --lines-per-shard")

    total_lines = count_lines(input_path)

    if args.lines_per_shard > 0:
        lines_per_shard = args.lines_per_shard
        num_shards = (total_lines + lines_per_shard - 1) // lines_per_shard
    else:
        num_shards = args.num_shards
        lines_per_shard = (total_lines + num_shards - 1) // num_shards

    print(f"[INFO] total_lines={total_lines}")
    print(f"[INFO] num_shards={num_shards}")
    print(f"[INFO] lines_per_shard={lines_per_shard}")

    shard_idx = 0
    lines_in_current = 0
    out_f = None
    shard_counts = []

    try:
        with input_path.open("r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                if out_f is None or lines_in_current >= lines_per_shard:
                    if out_f is not None:
                        out_f.close()
                        shard_counts.append(lines_in_current)
                    shard_path = output_dir / f"{args.prefix}_{shard_idx:04d}.jsonl"
                    out_f = shard_path.open("w", encoding="utf-8")
                    print(f"[INFO] writing {shard_path}")
                    shard_idx += 1
                    lines_in_current = 0

                out_f.write(line)
                lines_in_current += 1

        if out_f is not None:
            out_f.close()
            shard_counts.append(lines_in_current)

    finally:
        if out_f is not None and not out_f.closed:
            out_f.close()

    meta = {
        "input": str(input_path),
        "total_lines": total_lines,
        "num_shards": len(shard_counts),
        "lines_per_shard_target": lines_per_shard,
        "shard_counts": shard_counts,
        "prefix": args.prefix,
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote {len(shard_counts)} shards")


if __name__ == "__main__":
    main()
