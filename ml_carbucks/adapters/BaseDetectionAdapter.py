from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


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
