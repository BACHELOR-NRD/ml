from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


# @dataclass
# class BaseDetectionAdapter(ABC):
#     """
#     Base class for detection model adapters.
#     Handles:
#       - Model creation/loading from weights
#       - Class names
#       - Device selection
#       - Hyperparameters storage
#       - Dataset paths awareness
#       - Standard interface for training and evaluation
#     """

#     def __init__(
#         self,
#         classes: List[str],
#         model_path: Optional[Path | str] = None,
#         device: Optional[str] = None,
#         hparams: Optional[Dict[str, Any]] = None,
#         datasets: Optional[
#             Dict[str, Path | str]
#         ] = None,  # e.g., {"train": "...", "val": "...", "test": "..."}
#         metadata: Optional[Dict[str, Any]] = None,
#     ):
#         self.model_path = Path(model_path) if model_path else None
#         self.classes = classes or []
#         self.device = device or ("cuda" if self._cuda_available() else "cpu")
#         self.hparams = hparams or {}
#         self.datasets = datasets or {}
#         self.model = None
#         self.metadata = metadata or {}

#         self.load_model()
#         self.setup()  # Subclass decides how to implement

#     # ------------------------
#     # Abstract methods every adapter must implement
#     # ------------------------
#     @abstractmethod
#     def load_model(self):
#         """Load model from weights or create a new model from scratch."""
#         pass

#     @abstractmethod
#     def setup(self):
#         """Initialize datasets using the provided paths."""
#         pass

#     @abstractmethod
#     def fit(self):
#         """Train the model on the training dataset."""
#         pass

#     @abstractmethod
#     def evaluate(self) -> Dict[str, float]:
#         """Evaluate the model on the designated dataset(s)."""
#         pass

#     @abstractmethod
#     def predict(self, images: Any) -> List[Dict[str, Any]]:
#         """
#         Run full inference pipeline on a single image (or batch if desired).
#         Must return standardized detections:
#             [{"bbox": [x1, y1, x2, y2], "score": float, "label": str}, ...]
#         """
#         pass

#     @abstractmethod
#     def save_model(self, dir: Path | str) -> Path:
#         """Save the model weights to the specified path."""
#         pass

#     @abstractmethod
#     def clone_with_params(self, params: Dict[str, Any]) -> "BaseDetectionAdapter":
#         """Create a new adapter instance with the given hyperparameters."""
#         pass

#     # ------------------------
#     # Optional helpers
#     # ------------------------
#     def to(self, device: str):
#         """Move model to a different device."""
#         self.device = device
#         if self.model:
#             self.model.to(device)

#     @staticmethod
#     def _cuda_available() -> bool:
#         try:
#             import torch

#             return torch.cuda.is_available()
#         except ImportError:
#             return False


@dataclass
class BaseDetectionAdapter(ABC):
    classes: List[str]
    metadata: Dict[str, Any]
    hparams: Dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None
    model: Any = field(init=False, default=None)

    def __post_init__(self):
        self.device = self.device or ("cuda" if self._cuda_available() else "cpu")
        self.model = None

        self.health_check()

    # ------------------------

    @abstractmethod
    def get_required_metadata_keys(self) -> List[str]:
        pass

    @abstractmethod
    def get_possible_hyper_keys(self) -> List[str]:
        pass

    @abstractmethod
    def setup(self) -> "BaseDetectionAdapter":
        pass

    @abstractmethod
    def fit(self) -> "BaseDetectionAdapter":
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
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
    def save(self, dir: Path | str, prefix: str = "") -> Path:
        """Save the model weights to the specified path."""
        pass

    @abstractmethod
    def clone(self) -> "BaseDetectionAdapter":
        """Create a new adapter instance."""
        pass

    # ------------------------

    def set_params(self, params: Dict[str, Any]) -> "BaseDetectionAdapter":
        """Set hyperparameters and return self for chaining."""
        if self.model is not None:
            raise ValueError("Cannot set parameters after model has been created.")

        possible_keys = self.get_possible_hyper_keys()
        for key, value in params.items():
            if key in possible_keys:
                self.hparams[key] = value
            else:
                raise ValueError(
                    f"Invalid hyperparameter key: {key} for adapter {self.__class__.__name__}"
                )
        return self

    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.hparams.get(key, default)

    def health_check(self):
        required_keys = self.get_required_metadata_keys()
        missing_keys = [key for key in required_keys if key not in self.metadata]
        if missing_keys:
            raise ValueError(
                f"Missing required metadata keys: {missing_keys} for adapter {self.__class__.__name__}"
            )

        possible_keys = self.get_possible_hyper_keys()
        invalid_keys = [key for key in self.hparams if key not in possible_keys]
        if invalid_keys:
            raise ValueError(
                f"Invalid hyperparameter keys: {invalid_keys} for adapter {self.__class__.__name__}"
            )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
