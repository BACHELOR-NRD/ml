#!/usr/bin/env python3
"""
Script to create cross-validation fold datasets with multilabel stratified k-fold split.

This script takes a source dataset directory containing images and COCO annotations,
and creates fold_1, fold_2, etc. subdirectories with train/val splits.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_carbucks.utils.cross_validation import create_crossval_fold_structure
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    # Configuration
    source_dataset_dir = Path(
        "/home/bachelor/ml-carbucks/data/final_carbucks/crossval<>source"
    )
    output_base_dir = Path(
        "/home/bachelor/ml-carbucks/data/final_carbucks/crossval<>output"
    )
    coco_annotations_file = "instances_crossval.json"
    n_splits = 5
    random_state = 42

    logger.info("Starting cross-validation fold creation")
    logger.info(f"Source dataset: {source_dataset_dir}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Number of folds: {n_splits}")

    create_crossval_fold_structure(
        source_dataset_dir=source_dataset_dir,
        output_base_dir=output_base_dir,
        coco_annotations_file=coco_annotations_file,
        n_splits=n_splits,
        random_state=random_state,
    )

    logger.info("Cross-validation fold creation complete!")


if __name__ == "__main__":
    main()
