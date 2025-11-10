#!/usr/bin/env python3
"""
Minimal YOLO training script that relies on the YoloUltralyticsAdapter used
throughout the project. All important paths and hyperparameters are defined as
constants below so you can launch training with:

    python scripts/train_yolo_adapter.py

The script trains on the balanced Carbucks dataset and evaluates on the
matching validation split. Adjust the constants as needed for other datasets or
hyperparameter sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml_carbucks.adapters.UltralyticsAdapter import YoloUltralyticsAdapter
from ml_carbucks.adapters.BaseDetectionAdapter import ADAPTER_METRICS
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "carbucks_balanced"

CLASSES = ["scratch", "dent", "crack"]
WEIGHTS = REPO_ROOT / "yolo11l.pt"

EPOCHS = 150
BATCH_SIZE = 8
IMG_SIZE = 1024
LR = 1e-3
OPTIMIZER = "auto"
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 42

PROJECT_DIR = REPO_ROOT / "results" / "yolo_runs"
RUN_NAME = "carbucks_balanced_yolo"
SAVE_CHECKPOINTS = True
USE_AUGMENTATIONS = True
VERBOSE = False

# ---------------------------------------------------------------------------


def build_adapter() -> YoloUltralyticsAdapter:
    adapter = YoloUltralyticsAdapter(
        classes=CLASSES,
        weights=WEIGHTS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        lr=LR,
        optimizer=OPTIMIZER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        project_dir=PROJECT_DIR,
        name=RUN_NAME,
        seed=SEED,
        training_save=SAVE_CHECKPOINTS,
        training_augmentations=USE_AUGMENTATIONS,
        verbose=VERBOSE,
    )
    return adapter


def main() -> None:
    train_dataset = (
        str(DATA_ROOT / "images" / "train"),
        str(DATA_ROOT / "instances_train_curated.json"),
    )
    val_dataset = (
        str(DATA_ROOT / "images" / "val"),
        str(DATA_ROOT / "instances_val_curated.json"),
    )

    adapter = build_adapter()
    logger.info(
        "Training YOLO adapter for %d epochs on %s",
        EPOCHS,
        train_dataset[0],
    )

    adapter.setup().fit([train_dataset])
    logger.info("Training finished, running evaluation...")

    metrics: ADAPTER_METRICS = adapter.evaluate([val_dataset])
    pretty = json.dumps(metrics, indent=2)
    logger.info("Evaluation metrics:\n%s", pretty)
    print(pretty)


if __name__ == "__main__":
    main()
