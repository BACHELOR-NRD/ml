from dataclasses import dataclass
from typing import Any, Dict

import optuna


@dataclass
class TrialParamWrapper:
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
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "confidence_threshold": trial.suggest_float(
                "confidence_threshold", 0.1, 0.35
            ),
        }
        return params

    def get_param(self, trial: optuna.Trial, adapter_name: str) -> Dict[str, Any]:

        param_pairs = [
            ("yolo", self._get_ultralytics_params),
            ("rtdetr", self._get_ultralytics_params),
            ("fasterrcnn", self._get_fasterrcnn_params),
            ("efficientdet", self._get_efficientdet_params),
        ]

        for adapter_prefix, param_func in param_pairs:
            if adapter_name.lower().startswith(adapter_prefix):
                return param_func(trial)

        raise ValueError(f"Unknown adapter name: {adapter_name}")
