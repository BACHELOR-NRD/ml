from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Required
from dataclasses import dataclass, field
import numpy as np

import torch

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


class ADAPTER_PREDICTION(TypedDict):
    """Standardized per-image adapter prediction.

    Notes:
        - `boxes` is expected to be a 2D array of shape (N, 4) with [x1, y1, x2, y2].
        - `scores` is a 1D array of shape (N,) with confidence scores.
        - `labels` is a list of length N with class label indices.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


class ADAPTER_METRICS(TypedDict, total=False):
    """Standardized adapter evaluation metrics."""

    map_50: Required[float]
    map_75: Required[float]
    map_50_95: Required[float]
    classes: Required[List[int]]


ADAPTER_DATASETS = List[Tuple[str | Path, str | Path]]


@dataclass
class BaseDetectionAdapter(ABC):

    # --- HYPER PARAMETERS ---

    img_size: int = 256
    batch_size: int = 16
    epochs: int = 1

    # --- SETUP PARAMETERS ---

    weights: str | Path | dict = field(default="DEFAULT")
    device: str = field(init=False)
    model: Any = field(init=False, default=None)
    verbose: bool = field(default=False)

    def __post_init__(self):
        self.device = "cuda" if self._cuda_available() else "cpu"

    # ------------------------

    @abstractmethod
    def setup(self) -> "BaseDetectionAdapter":
        pass

    @abstractmethod
    def fit(self, datasets: ADAPTER_DATASETS) -> "BaseDetectionAdapter":
        pass

    @abstractmethod
    def evaluate(self, datasets: ADAPTER_DATASETS) -> ADAPTER_METRICS:
        pass

    @abstractmethod
    def predict(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.0,
        iou_threshold: float = 1.0,
        max_detections: int = 10,
    ) -> List[ADAPTER_PREDICTION]:
        """
        Run full inference pipeline on a single image (or batch if desired).
        Must return standardized detections:
            [{"bbox": [x1, y1, x2, y2], "score": float, "label": str}, ...]
        """
        pass

    @abstractmethod
    def debug(
        self,
        train_datasets: ADAPTER_DATASETS,
        val_datasets: ADAPTER_DATASETS,
        results_path: str | Path,
        results_name: str,
        visualize: Literal["every", "last", "none"] = "none",
    ) -> ADAPTER_METRICS:
        """Debug training and evaluation loops."""
        pass

    @abstractmethod
    def save_weights(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        """Save the model weights to the specified path."""
        pass

    @abstractmethod
    def save_pickled(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        """Save pickled model to the specified path."""
        pass

    @staticmethod
    @abstractmethod
    def load_pickled(path: str | Path) -> "BaseDetectionAdapter":
        """Load pickled model from the specified path."""
        pass

    # ------------------------

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def set_params(self, params: Dict[str, Any]) -> "BaseDetectionAdapter":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Parameter {key} not found in {self.__class__.__name__}"
                )
        return self

    def get_params(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["model", "device"]
        }

    def clone(self) -> "BaseDetectionAdapter":
        """Create a new adapter instance with the same parameters."""
        cls = self.__class__
        return cls(**self.get_params())
