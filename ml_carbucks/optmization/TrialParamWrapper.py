from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from typing_extensions import Literal

import optuna

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrialParamWrapper:
    # fmt: off
    version: Literal["h1", "h2", "e1", "e2", "e3"]
    ensemble_size: Optional[int] = None
    """A class that is to help the creation of the trial parameters."""

    V1_IMG_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [384])
    """Smaller sizes for faster experiments."""

    V2_IMG_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [768])
    """Bigger sizes for better accuracy."""

    def _get_yolo_params(self, trial: optuna.Trial) -> Dict[str, Any]:

        params: Dict[str, Any] = {
            "epochs": trial.suggest_int("epochs", 15, 40),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
            "momentum": trial.suggest_float("momentum", 0.3, 0.99),
        }

        if self.version not in ("h1", "h2"):
            raise ValueError(
                f"YOLO parameters are only available for versions 'h1' and 'h2', got '{self.version}'"
            )

        elif self.version == "h1":
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["yolo11m.pt"]),
                    "img_size": trial.suggest_categorical("img_size", self.V1_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [1]),
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )
        else:
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["yolo11x.pt"]),
                    "img_size": trial.suggest_categorical("img_size", self.V2_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [8]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [2, 4]),
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )

        return params

    def _get_rtdetr_params(self, trial: optuna.Trial) -> Dict[str, Any]:

        params: Dict[str, Any] = {
            "epochs": trial.suggest_int("epochs", 15, 40),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "momentum": trial.suggest_float("momentum", 0.3, 0.99),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        }

        if self.version not in ("h1", "h2"):
            raise ValueError(
                f"RTDETR parameters are only available for versions 'h1' and 'h2', got '{self.version}'"
            )

        elif self.version == "h1":
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["rtdetr-l.pt"]),
                    "img_size": trial.suggest_categorical("img_size", self.V1_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [1]),
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )
        else:
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["rtdetr-x.pt"]),
                    "img_size": trial.suggest_categorical("img_size", self.V2_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [8]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [2, 4]),
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )
        return params

    def _get_fasterrcnn_params(self, trial: optuna.Trial) -> Dict[str, Any]:

        params: Dict[str, Any] = {
            "epochs": trial.suggest_int("epochs", 15, 40),
            "lr_head": trial.suggest_float("lr_head", 5e-5, 3e-3, log=True),
            "weight_decay_head": trial.suggest_float("weight_decay_head", 1e-5, 1e-3, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["SGD", "AdamW"]),
            # "clip_gradients": trial.suggest_categorical("clip_gradients", [None, 0.5, 2.0]),

        }

        if self.version not in ("h1", "h2"):
            raise ValueError(
                f"FasterRCNN parameters are only available for versions 'h1' and 'h2', got '{self.version}'"
            )

        elif self.version == "h1":
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["V1"]),
                    "img_size": trial.suggest_categorical("img_size", self.V1_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [1]),
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )
        else:
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["V2"]),
                    "img_size": trial.suggest_categorical("img_size", self.V2_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [8]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [2]),  # NOTE: should also include 4?
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )

        return params

    def _get_efficientdet_params(self, trial: optuna.Trial) -> Dict[str, Any]:

        params: Dict[str, Any] = {
            "epochs": trial.suggest_int("epochs", 15, 40),
            "lr": trial.suggest_float("lr", 5e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-6, 1e-2, log=True),
            "loader": trial.suggest_categorical("loader", ["inbuilt"]),
            "optimizer": trial.suggest_categorical("optimizer", ["momentum", "adamw"]),
        }

        if self.version not in ("h1", "h2"):
            raise ValueError(
                f"EfficientDet parameters are only available for versions 'h1' and 'h2', got '{self.version}'"
            )

        elif self.version == "h1":
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["tf_efficientdet_d0"]),
                    "img_size": trial.suggest_categorical("img_size", self.V1_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [1]),
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )
        else:
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["tf_efficientdet_d3"]),
                    "img_size": trial.suggest_categorical("img_size", self.V2_IMG_SIZE_OPTIONS),
                    "batch_size": trial.suggest_categorical("batch_size", [4]),
                    "accumulation_steps": trial.suggest_categorical("accumulation_steps", [4, 8]),
                    "scheduler": trial.suggest_categorical("scheduler", [None]),
                }
            )

        return params

    def _get_ensemble_model_params(self, trial: optuna.Trial) -> Dict[str, Any]:

        if self.version not in ["e1", "e2", "e3"]:
            raise ValueError(
                f"Ensemble model parameters are only available for versions 'v3' and 'v4', got '{self.version}'"
            )

        elif self.version == "e1":
            params = {
                "fusion_strategy": trial.suggest_categorical("fusion_strategy", ["nms"]),
                "fusion_conf_threshold": trial.suggest_float("fusion_conf_threshold", 0.05, 0.60),
                "fusion_iou_threshold": trial.suggest_float("fusion_iou_threshold", 0.35, 0.75),
                "fusion_max_detections": trial.suggest_int("fusion_max_detections", 5, 5),
                "fusion_norm_method": trial.suggest_categorical("fusion_norm_method", ["zscore"]),
            }

        elif self.version == "e2":

            params = {
                "fusion_strategy": trial.suggest_categorical("fusion_strategy", ["wbf"]),
                "fusion_conf_threshold": trial.suggest_float("fusion_conf_threshold", 0.05, 0.60),
                "fusion_iou_threshold": trial.suggest_float("fusion_iou_threshold", 0.45, 0.75),
                "fusion_max_detections": trial.suggest_int("fusion_max_detections", 5, 5),
                "fusion_norm_method": trial.suggest_categorical("fusion_norm_method", ["zscore"]),
            }
        else:

            params = {
                "fusion_strategy": trial.suggest_categorical("fusion_strategy", ["nms", "wbf"]),
                "fusion_conf_threshold": trial.suggest_float("fusion_conf_threshold", 0.00, 0.60),
                "fusion_iou_threshold": trial.suggest_float("fusion_iou_threshold", 0.35, 0.75),
                "fusion_max_detections": trial.suggest_int("fusion_max_detections", 5, 5),
                "fusion_norm_method": trial.suggest_categorical("fusion_norm_method", ["zscore", "minmax", "quantile"]),
            }

        if self.ensemble_size is None:

            logger.warning("Ensemble size not provided in kwargs; cannot suggest trust weights.")
            params["fusion_trust_factors"] = trial.suggest_categorical("fusion_trust_factors", [None])
            params["fusion_exponent_factors"] = trial.suggest_categorical("fusion_exponent_factors", [None])
        else:

            for i in range(self.ensemble_size):
                params[f"fusion_trust_factor_{i}"] = trial.suggest_float(f"fusion_trust_factor_{i}", 0.8, 1.0)
                if self.version == "e2" or (self.version == "e3" and params["fusion_strategy"] == "wbf"):
                    params[f"fusion_exponent_factor_{i}"] = trial.suggest_float(f"fusion_exponent_factor_{i}", 0.8, 1.2)

        final_params = self.convert_ensemble_params_to_model_format(
            params, ensemble_size=self.ensemble_size
        )

        return final_params

    @staticmethod
    def convert_ensemble_params_to_model_format(
        params: Dict[str, Any], ensemble_size: Optional[int] = None
    ) -> Dict[str, Any]:
        final_params = {k: v for k, v in params.items() if "factor_" not in k}

        if ensemble_size is None:
            final_params["fusion_trust_factors"] = None
            final_params["fusion_exponent_factors"] = None
        else:

            if "fusion_trust_factor_0" in params:
                final_params["fusion_trust_factors"] = [params[f"fusion_trust_factor_{i}"] for i in range(ensemble_size)]
            else:
                final_params["fusion_trust_factors"] = None

            if "fusion_exponent_factor_0" in params:
                final_params["fusion_exponent_factors"] = [params[f"fusion_exponent_factor_{i}"] for i in range(ensemble_size)]
            else:
                final_params["fusion_exponent_factors"] = None

        return final_params

    # fmt: on
    def get_param(
        self,
        trial: optuna.Trial,
        adapter_name: str,
    ) -> Dict[str, Any]:

        param_pairs = [
            ("yolo", self._get_yolo_params),
            ("rtdetr", self._get_rtdetr_params),
            ("fasterrcnn", self._get_fasterrcnn_params),
            ("efficientdet", self._get_efficientdet_params),
            ("ensemblemodel", self._get_ensemble_model_params),
        ]

        for adapter_prefix, param_func in param_pairs:
            if adapter_name.lower().startswith(adapter_prefix):
                return param_func(trial)

        raise ValueError(f"Unknown adapter name: {adapter_name}")
