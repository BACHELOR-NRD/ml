from dataclasses import dataclass
from typing import Any, Dict, Optional

import optuna

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrialParamWrapper:
    kwargs: Optional[Dict[str, Any]] = None
    """A class that is to help the creation of the trial parameters."""

    IMG_SIZE_OPTIONS = [
        # 256,
        384,
        # 512,
        # 640,
        # 768,
        # 1024,
    ]

    def _get_ultralytics_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "img_size": trial.suggest_categorical("img_size", self.IMG_SIZE_OPTIONS),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_int("epochs", 10, 30),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.3, 0.99),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["NAdam", "AdamW"]),
        }
        return params

    def _get_fasterrcnn_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "img_size": trial.suggest_categorical("img_size", self.IMG_SIZE_OPTIONS),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "epochs": trial.suggest_int("epochs", 10, 30),
            "lr_backbone": trial.suggest_float("lr_backbone", 1e-5, 5e-4, log=True),
            "lr_head": trial.suggest_float("lr_head", 5e-5, 3e-3, log=True),
            "weight_decay_backbone": trial.suggest_float(
                "weight_decay_backbone", 1e-6, 1e-3, log=True
            ),
            "weight_decay_head": trial.suggest_float(
                "weight_decay_head", 1e-5, 1e-3, log=True
            ),
        }
        return params

    def _get_efficientdet_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "img_size": trial.suggest_categorical("img_size", self.IMG_SIZE_OPTIONS),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_int("epochs", 10, 30),
            "optimizer": trial.suggest_categorical("optimizer", ["momentum", "adam"]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-6, 1e-2, log=True),
            "loader": trial.suggest_categorical("loader", ["inbuild", "custom"]),
        }
        return params

    def _get_ensemble_model_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "fusion_strategy": trial.suggest_categorical(
                "fusion_strategy", ["nms", "wbf"]
            ),
            "fusion_conf_threshold": trial.suggest_float(
                "fusion_conf_threshold", 0.01, 0.5
            ),
            "fusion_iou_threshold": trial.suggest_float(
                "fusion_iou_threshold", 0.2, 0.8
            ),
            "fusion_max_detections": trial.suggest_int("fusion_max_detections", 5, 10),
            "fusion_norm_method": trial.suggest_categorical(
                "fusion_norm_method", ["minmax", "zscore", None]
            ),
        }

        ensemble_size = (
            self.kwargs.get("ensemble_size", None) if self.kwargs is not None else None
        )
        if ensemble_size is None:
            logger.warning(
                "Ensemble size not provided in kwargs; cannot suggest trust weights."
            )
            params["fusion_trust_weights"] = None
        else:
            trust_weights = []
            for i in range(ensemble_size):
                weight = trial.suggest_float(f"trust_weight_{i}", 0.0, 1.0)
                trust_weights.append(weight)
            params["fusion_trust_weights"] = trust_weights

        return params

    def get_param(self, trial: optuna.Trial, adapter_name: str) -> Dict[str, Any]:

        param_pairs = [
            ("yolo", self._get_ultralytics_params),
            ("rtdetr", self._get_ultralytics_params),
            ("fasterrcnn", self._get_fasterrcnn_params),
            ("efficientdet", self._get_efficientdet_params),
            ("ensemblemodel", self._get_ensemble_model_params),
        ]

        for adapter_prefix, param_func in param_pairs:
            if adapter_name.lower().startswith(adapter_prefix):
                return param_func(trial)

        raise ValueError(f"Unknown adapter name: {adapter_name}")
