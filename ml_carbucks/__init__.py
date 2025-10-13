import os
from pathlib import Path


RESULTS_DIR = Path(os.path.abspath(__file__)).parent.parent / "results"

DATA_DIR = Path(os.path.abspath(__file__)).parent.parent / "data"
DATA_CAR_DD_DIR = DATA_DIR / "car_dd"
DATA_CAR_DD_YAML = DATA_CAR_DD_DIR / "dataset.yaml"
DATA_CAR_DD_TEST_IMAGES_DIR = DATA_CAR_DD_DIR / "images" / "test"

YOLO_PRETRAINED_11N = DATA_DIR / "ultralytics_models" / "yolo11n.pt"
YOLO_PRETRAINED_11S = DATA_DIR / "ultralytics_models" / "yolo11s.pt"
YOLO_PRETRAINED_11M = DATA_DIR / "ultralytics_models" / "yolo11m.pt"
YOLO_PRETRAINED_11L = DATA_DIR / "ultralytics_models" / "yolo11l.pt"
YOLO_PRETRAINED_11X = DATA_DIR / "ultralytics_models" / "yolo11x.pt"

RTDETR_PRETRAINED_L = DATA_DIR / "ultralytics_models" / "rtdetr-l.pt"


os.makedirs(RESULTS_DIR, exist_ok=True)

# Explicit public API for `from ml_carbucks import *` and clarity when importing names
__all__ = [
    "RESULTS_DIR",
    "DATA_DIR",
    "DATA_CAR_DD_DIR",
    "DATA_CAR_DD_YAML",
    "DATA_CAR_DD_TEST_IMAGES_DIR",
    "YOLO_PRETRAINED_11N",
    "YOLO_PRETRAINED_11S",
    "YOLO_PRETRAINED_11M",
    "YOLO_PRETRAINED_11L",
    "YOLO_PRETRAINED_11X",
    "RTDETR_PRETRAINED_L",
]
