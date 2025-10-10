#!/usr/bin/env python3
"""
Copy images referenced by label files from a source image directory to a destination directory.

Defaults (as requested):
- labels_dir: /home/bachelor/ml-carbucks/data/carbucks/labels/picsa-m_job1
- source_images_dir: /home/mainnural/shared/picsa-m
- dest_images_dir: /home/bachelor/ml-carbucks/data/carbucks/images/pica-m_job1

Usage:
  python3 scripts/copy_images_for_labels.py --dry-run
  python3 scripts/copy_images_for_labels.py --labels-dir /path/to/labels --source /path/to/images --dest /path/to/dest

Features:
- Supports common image extensions (.jpg, .jpeg, .png, .bmp, .tif, .tiff)
- If multiple files with the same basename exist in source, picks the first found (prints a warning)
- Creates destination directory if missing
- Dry-run mode to preview copy operations
- Reports missing images
"""

from pathlib import Path
import shutil
import argparse
from typing import List, Set

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_label_basenames(labels_dir: Path) -> List[str]:
    basenames: Set[str] = set()
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")
    for f in labels_dir.iterdir():
        if f.is_file() and f.suffix == ".txt":
            # label files named like <basename>.txt
            basenames.add(f.stem)
    return sorted(basenames)


def find_source_for_basename(source_dir: Path, basename: str) -> List[Path]:
    found = []
    for ext in IMAGE_EXTS:
        candidate = source_dir / (basename + ext)
        if candidate.exists():
            found.append(candidate)
    # also try case-insensitive scan (slower) if nothing found
    if not found and source_dir.exists():
        # scan directory entries (non-recursive)
        for p in source_dir.iterdir():
            if (
                p.is_file()
                and p.stem.lower() == basename.lower()
                and p.suffix.lower() in IMAGE_EXTS
            ):
                found.append(p)
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels-dir",
        type=str,
        default="/home/bachelor/ml-carbucks/data/carbucks/labels/pics0-9_job40",
    )
    p.add_argument("--source", type=str, default="/home/mainnural/shared/pics0-9")
    p.add_argument(
        "--dest",
        type=str,
        default="/home/bachelor/ml-carbucks/data/carbucks/images/pics0-9_job40",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search source dir recursively if not found directly",
    )
    args = p.parse_args()

    labels_dir = Path(args.labels_dir)
    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    basenames = list_label_basenames(labels_dir)
    print(f"Found {len(basenames)} label basenames in {labels_dir}")

    if not source_dir.exists():
        raise SystemExit(f"Source images dir not found: {source_dir}")

    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    copied = 0
    ambiguous = []

    for b in basenames:
        candidates = find_source_for_basename(source_dir, b)
        if not candidates and args.recursive:
            # recursive search
            for p in source_dir.rglob("*"):
                if (
                    p.is_file()
                    and p.stem.lower() == b.lower()
                    and p.suffix.lower() in IMAGE_EXTS
                ):
                    candidates.append(p)
        if not candidates:
            missing.append(b)
            continue
        if len(candidates) > 1:
            ambiguous.append((b, candidates))
        src = candidates[0]
        dst = dest_dir / src.name
        print(f"Copy: {src} -> {dst}")
        if not args.dry_run:
            shutil.copy2(src, dst)
        copied += 1

    print(f"Copied: {copied} images")
    if missing:
        print(f"Missing {len(missing)} images (no files found for these basenames):")
        for m in missing[:50]:
            print("-", m)
    if ambiguous:
        print(
            f"{len(ambiguous)} basenames had ambiguous matches (multiple candidate files in source):"
        )
        for b, cands in ambiguous[:20]:
            print("-", b, "->", [str(x) for x in cands[:5]])


if __name__ == "__main__":
    main()
