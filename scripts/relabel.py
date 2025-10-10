#!/usr/bin/env python3
"""
Script: relabel_car_dd.py
Purpose: Update dataset.yaml `names` mapping and remap class indices inside YOLO label .txt files

Usage examples:
  # swap class 0 and 1
  python scripts/relabel_car_dd.py --map 0:1,1:0 --dataset data/car_dd/dataset.yaml --labels-dir data/car_dd/labels

  # provide new names list (will overwrite names in dataset.yaml). The list length defines nc.
  python scripts/relabel_car_dd.py --new-names dent,scratch,crack --dataset data/car_dd/dataset.yaml --labels-dir data/car_dd/labels

  # dry-run (do not write any files)
  python scripts/relabel_car_dd.py --map 0:2,1:0 --dry-run

This script will create a backup of dataset.yaml as dataset.yaml.bak.TIMESTAMP and optionally create a backup folder for labels if not dry-run.
"""

from __future__ import annotations

import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def read_dataset_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_dataset_yaml(path: Path, data: Dict, backup: bool = True):
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak.{stamp}")
        shutil.copy(path, bak)
        print(f"Backed up {path} to {bak}")
    with path.open("w") as f:
        yaml.dump(data, f, sort_keys=False)
        print(f"Wrote updated dataset yaml to {path}")


def parse_map(map_str: str) -> Dict[int, int]:
    """Parse mapping like '0:1,1:0,2:2' into dict{int:int}"""
    m: Dict[int, int] = {}
    if not map_str:
        return m
    for part in map_str.split(","):
        if ":" not in part:
            raise ValueError(f"Bad mapping token: {part}")
        a, b = part.split(":", 1)
        m[int(a.strip())] = int(b.strip())
    return m


def build_permutation_from_new_names(
    old_names: Dict[int, str], new_names_list: List[str]
) -> Dict[int, int]:
    """Attempt to build a mapping from old indices to new indices by matching names.
    If matching fails for some names, unmapped indices will be assigned to themselves or to -1.
    """
    new_map: Dict[int, int] = {}
    # create name->index for new names
    new_name_to_idx = {name: i for i, name in enumerate(new_names_list)}
    for old_idx, old_name in old_names.items():
        if old_name in new_name_to_idx:
            new_map[old_idx] = new_name_to_idx[old_name]
        else:
            # keep same index if possible
            if old_idx < len(new_names_list):
                new_map[old_idx] = old_idx
            else:
                # assign to 0 fallback
                new_map[old_idx] = 0
    return new_map


def remap_label_file(path: Path, mapping: Dict[int, int]) -> int:
    """Read a YOLO label file and remap class indices according to mapping.
    Returns number of lines changed.
    """
    changed = 0
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    new_lines = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            new_lines.append(ln)
            continue
        parts = ln.split()
        try:
            cls = int(float(parts[0]))
        except Exception:
            print(f"Skipping unparsable line in {path}: {ln}")
            new_lines.append(ln)
            continue
        new_cls = mapping.get(cls, cls)
        if new_cls != cls:
            changed += 1
        # rewrite line with same bbox numbers
        new_lines.append(" ".join([str(new_cls)] + parts[1:]))
    path.write_text("\n".join(new_lines))
    return changed


def find_label_files(labels_dir: Path) -> List[Path]:
    files: List[Path] = []
    for sub in ["train", "val", "test"]:
        p = labels_dir / sub
        if p.exists() and p.is_dir():
            files.extend(sorted(p.glob("*.txt")))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="data/car_dd/dataset.yaml",
        help="Path to dataset.yaml",
    )
    p.add_argument(
        "--labels-dir",
        type=str,
        default="data/car_dd/labels",
        help="Path to labels folder containing train/val/test subfolders",
    )
    p.add_argument(
        "--map",
        type=str,
        default="",
        help="Comma-separated mapping old:new e.g. '0:1,1:0'",
    )
    p.add_argument(
        "--new-names",
        type=str,
        default="",
        help="Comma-separated new names list e.g. 'dent,scratch,crack'",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, only show what would be done",
    )
    p.add_argument(
        "--backup-labels",
        action="store_true",
        help="Backup labels folder before modifying",
    )
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    labels_dir = Path(args.labels_dir)

    if not dataset_path.exists():
        raise SystemExit(f"dataset.yaml not found: {dataset_path}")

    data = read_dataset_yaml(dataset_path)
    print("Current dataset.yaml names:", data.get("names"))

    mapping: Dict[int, int] = {}

    if args.new_names:
        new_names_list = [n.strip() for n in args.new_names.split(",") if n.strip()]
        # build mapping based on names if possible
        old_names = data.get("names", {})
        mapping = build_permutation_from_new_names(old_names, new_names_list)
        # update data.names to numeric-keyed map required in this repo format
        new_names_map = {i: name for i, name in enumerate(new_names_list)}
        data["names"] = new_names_map
        data["nc"] = len(new_names_list)
        print("Will replace dataset names with:", new_names_map)
    elif args.map:
        mapping = parse_map(args.map)
        print("Will apply mapping:", mapping)
    else:
        raise SystemExit("Either --map or --new-names must be provided")

    label_files = find_label_files(labels_dir)
    print(f"Found {len(label_files)} label files to process")

    if args.dry_run:
        print(
            "Dry-run mode: no files will be written. Showing sample remapping for first 10 files..."
        )
        sample = label_files[:10]
        for f in sample:
            txt = f.read_text().splitlines()
            print("---", f)
            for ln in txt[:5]:
                print(ln)
            print("---")
        print("Mapping to apply:", mapping)
        return

    # backup dataset.yaml
    write_dataset_yaml(dataset_path, data, backup=True)

    # optional backup labels
    if args.backup_labels:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bakdir = labels_dir.parent / f"{labels_dir.name}_bak_{stamp}"
        shutil.copytree(labels_dir, bakdir)
        print(f"Backed up labels directory to {bakdir}")

    total_changes = 0
    for f in label_files:
        changed = remap_label_file(f, mapping)
        if changed:
            print(f"Updated {f}: {changed} lines changed")
            total_changes += changed

    print(f"Total label lines changed: {total_changes}")


if __name__ == "__main__":
    main()
