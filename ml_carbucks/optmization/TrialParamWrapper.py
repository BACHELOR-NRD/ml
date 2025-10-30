from dataclasses import dataclass
from typing import Any, Dict

import optuna


@dataclass
class TrialParamWrapper:
    """A class that is to help the creation of the trial parameters."""

    def _get_ultralytics_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "lr0": trial.suggest_float("lr0", 1e-5, 1e-2, log=True),
            "momentum": trial.suggest_float("momentum", 0.6, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        }
        return params

    def _get_fasterrcnn_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "momentum": trial.suggest_float("momentum", 0.6, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        }
        return params

    def _get_efficientdet_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "momentum": trial.suggest_float("momentum", 0.6, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        }
        return params

    def get_param(self, trial: optuna.Trial, adapter_name: str) -> Dict[str, Any]:

        param_pairs = [
            ("ultralytics", self._get_ultralytics_params),
            ("fasterrcnn", self._get_fasterrcnn_params),
            ("efficientdet", self._get_efficientdet_params),
        ]

        for adapter_prefix, param_func in param_pairs:
            if adapter_name.lower().startswith(adapter_prefix):
                return param_func(trial)

        raise ValueError(f"Unknown adapter name: {adapter_name}")
