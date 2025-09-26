import os
from pathlib import Path


RESULTS_DIR = Path(os.path.abspath(__file__)).parent.parent / "results"

DATA_CAR_DD_DIR = Path(os.path.abspath(__file__)).parent.parent / "data" / "car_dd"
DATA_CAR_DD_YAML = DATA_CAR_DD_DIR / "dataset.yaml"
DATA_CAR_DD_TEST_IMAGES_DIR = DATA_CAR_DD_DIR / "images" / "test"
