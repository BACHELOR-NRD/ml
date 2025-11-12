import os
from pathlib import Path


DATA_DIR = Path(os.path.abspath(__file__)).parent.parent / "data"
TEST_DIR = Path(os.path.abspath(__file__)).parent.parent / "tests"
RESULTS_DIR = Path(os.path.abspath(__file__)).parent.parent / "results"
PRODUCTS_DIR = Path(os.path.abspath(__file__)).parent.parent / "products"
OPTUNA_DIR = Path(os.path.abspath(__file__)).parent.parent / "optuna"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRODUCTS_DIR, exist_ok=True)
os.makedirs(OPTUNA_DIR, exist_ok=True)

# Explicit public API for `from ml_carbucks import *` and clarity when importing names

__all__ = [
    "RESULTS_DIR",
    "DATA_DIR",
    "TEST_DIR",
    "PRODUCTS_DIR",
]
