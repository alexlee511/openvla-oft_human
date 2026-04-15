#!/usr/bin/env python3
"""Patch finetune.py to switch between bfloat16 and float16.

Usage:
    python vla-scripts/patch_dtype.py --to float16    # for GPUs without bfloat16 (T4, V100, etc.)
    python vla-scripts/patch_dtype.py --to bfloat16   # revert to original (A100, H100, etc.)
"""

import argparse
import re
from pathlib import Path


def patch(filepath: Path, src_dtype: str, dst_dtype: str) -> int:
    text = filepath.read_text()
    new_text = text.replace(f"torch.{src_dtype}", f"torch.{dst_dtype}")
    count = text.count(f"torch.{src_dtype}")
    if count == 0:
        print(f"No occurrences of torch.{src_dtype} found — file may already use torch.{dst_dtype}.")
        return 0
    filepath.write_text(new_text)
    print(f"Replaced {count} occurrences of torch.{src_dtype} → torch.{dst_dtype} in {filepath}")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--to", required=True, choices=["float16", "bfloat16"], dest="target",
                        help="Target dtype to switch to")
    parser.add_argument("--file", default=None,
                        help="Path to finetune.py (default: vla-scripts/finetune.py next to this script)")
    args = parser.parse_args()

    if args.file:
        filepath = Path(args.file)
    else:
        filepath = Path(__file__).parent / "finetune.py"

    if not filepath.exists():
        parser.error(f"File not found: {filepath}")

    src = "bfloat16" if args.target == "float16" else "float16"
    patch(filepath, src, args.target)


if __name__ == "__main__":
    main()
