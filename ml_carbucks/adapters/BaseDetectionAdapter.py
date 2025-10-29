from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseDetectionAdapter(ABC):
    """
    Base class for detection model adapters.
    Handles:
      - Model creation/loading from weights
      - Class names
      - Device selection
      - Hyperparameters storage
      - Dataset paths awareness
      - Standard interface for training and evaluation
    """

    def __init__(
        self,
        classes: List[str],
        model_path: Optional[Path | str] = None,
        device: Optional[str] = None,
        hparams: Optional[Dict[str, Any]] = None,
        datasets: Optional[
            Dict[str, Path | str]
        ] = None,  # e.g., {"train": "...", "val": "...", "test": "..."}
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.classes = classes or []
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.hparams = hparams or {}
        self.datasets = datasets or {}
        self.model = None
        self.metadata = metadata or {}

        self.load_model()
        self.setup()  # Subclass decides how to implement

    # ------------------------
    # Abstract methods every adapter must implement
    # ------------------------
    @abstractmethod
    def load_model(self):
        """Load model from weights or create a new model from scratch."""
        pass

    @abstractmethod
    def setup(self):
        """Initialize datasets using the provided paths."""
        pass

    @abstractmethod
    def fit(self):
        """Train the model on the training dataset."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the designated dataset(s)."""
        pass

    @abstractmethod
    def predict(self, images: Any) -> List[Dict[str, Any]]:
        """
        Run full inference pipeline on a single image (or batch if desired).
        Must return standardized detections:
            [{"bbox": [x1, y1, x2, y2], "score": float, "label": str}, ...]
        """
        pass

    @abstractmethod
    def save_model(self, save_path: Path | str):
        """Save the model weights to the specified path."""
        pass

    # ------------------------
    # Optional helpers
    # ------------------------
    def to(self, device: str):
        """Move model to a different device."""
        self.device = device
        if self.model:
            self.model.to(device)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
