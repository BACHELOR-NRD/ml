import os
from pathlib import Path


RESULTS_DIR = Path(os.path.abspath(__file__)).parent.parent / "results"

DATA_DIR = Path(os.path.abspath(__file__)).parent.parent / "data"
DATA_CAR_DD_DIR = DATA_DIR / "car_dd"
DATA_CAR_DD_YAML = DATA_CAR_DD_DIR / "dataset.yaml"
DATA_CAR_DD_TEST_IMAGES_DIR = DATA_CAR_DD_DIR / "images" / "test"

YOLO_PRETRAINED_11L = DATA_DIR / "yolo_models" / "yolo11l.pt"
YOLO_PRETRAINED_11N = DATA_DIR / "yolo_models" / "yolo11n.pt"


os.makedirs(RESULTS_DIR, exist_ok=True)
