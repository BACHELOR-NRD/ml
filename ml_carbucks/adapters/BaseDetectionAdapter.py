from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, TypedDict
from dataclasses import dataclass, field

from numpy.typing import NDArray
import torch

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


class ADAPTER_PREDICTION(TypedDict):
    """Standardized per-image adapter prediction.

    Notes:
        - `boxes` is expected to be a 2D array of shape (N, 4) with [x1, y1, x2, y2].
        - `scores` is a 1D array of shape (N,) with confidence scores.
        - `labels` is a list of length N with string labels (or category names).
        - `image_ids` can be used when batching predictions from multiple images.
    """

    boxes: NDArray
    scores: NDArray
    labels: List[str]
    image_ids: List[int]


@dataclass
class BaseDetectionAdapter(ABC):
    classes: List[str]
    weights: str | Path

    img_size: int = 256
    batch_size: int = 16
    epochs: int = 1

    device: str = field(init=False)
    model: Any = field(init=False, default=None)

    def __post_init__(self):
        self.device = "cuda" if self._cuda_available() else "cpu"

    # ------------------------

    @abstractmethod
    def setup(self) -> "BaseDetectionAdapter":
        pass

    @abstractmethod
    def fit(self, img_dir: str | Path, ann_file: str | Path) -> "BaseDetectionAdapter":
        pass

    @abstractmethod
    def evaluate(self, img_dir: str | Path, ann_file: str | Path) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, images: List[torch.Tensor]) -> List[ADAPTER_PREDICTION]:
        """
        Run full inference pipeline on a single image (or batch if desired).
        Must return standardized detections:
            [{"bbox": [x1, y1, x2, y2], "score": float, "label": str}, ...]
        """
        pass

    @abstractmethod
    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        """Save the model weights to the specified path."""
        pass

    @abstractmethod
    def clone(self) -> "BaseDetectionAdapter":
        """Create a new adapter instance."""
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
