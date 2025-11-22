from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional
from typing_extensions import Literal

import optuna

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrialParamWrapper:
    version: Literal["v1", "v2"] = "v1"
    ensemble_size: Optional[int] = None
    """A class that is to help the creation of the trial parameters."""

    V1_IMG_SIZE_OPTIONS: List[int] = [384]  # NOTE: smaller sizes for faster experiments

    V2_IMG_SIZE_OPTIONS: List[int] = [1024]

    def _get_ultralytics_params(self, trial: optuna.Trial, name: str) -> Dict[str, Any]:

        params: Dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }
        if self.version == "v1":
            params.update(
                {
                    "weights": trial.suggest_categorical(
                        "weights", ["yolo11m.pt" if name == "yolo" else "rtdetr-l.pt"]
                    ),
                    "epochs": trial.suggest_int("epochs", 10, 30),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                    "img_size": trial.suggest_categorical(
                        "img_size", self.V1_IMG_SIZE_OPTIONS
                    ),
                    "momentum": trial.suggest_float("momentum", 0.3, 0.99),
                    "optimizer": trial.suggest_categorical(
                        "optimizer", ["Adam", "AdamW"]
                    ),
                }
            )
        if self.version == "v2":
            params.update(
                {
                    "weights": trial.suggest_categorical(
                        "weights", ["yolo11x.pt" if name == "yolo" else "rtdetr-x.pt"]
                    ),
                    "epochs": trial.suggest_int("epochs", 20, 60),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                    "img_size": trial.suggest_categorical(
                        "img_size", self.V2_IMG_SIZE_OPTIONS
                    ),
                    "momentum": trial.suggest_float("momentum", 0.3, 0.99),
                    "optimizer": trial.suggest_categorical(
                        "optimizer", ["Adam", "AdamW"]
                    ),
                }
            )
        return params

    def _get_fasterrcnn_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "lr_head": trial.suggest_float("lr_head", 5e-5, 3e-3, log=True),
            "weight_decay_head": trial.suggest_float(
                "weight_decay_head", 1e-5, 1e-3, log=True
            ),
            # "clip_gradients": trial.suggest_categorical(
            #     "clip_gradients", [None, 0.5, 2.0]
            # ),
        }

        if self.version == "v1":
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["V1"]),
                    "epochs": trial.suggest_int("epochs", 10, 30),
                    "img_size": trial.suggest_categorical(
                        "img_size", self.V1_IMG_SIZE_OPTIONS
                    ),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 13]),
                    "optimizer": trial.suggest_categorical(
                        "optimizer", ["SGD", "AdamW"]
                    ),
                }
            )
        if self.version == "v2":
            params.update(
                {
                    "weights": trial.suggest_categorical("weights", ["V2"]),
                    "epochs": trial.suggest_int("epochs", 20, 60),
                    "img_size": trial.suggest_categorical(
                        "img_size", self.V2_IMG_SIZE_OPTIONS
                    ),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                    "optimizer": trial.suggest_categorical(
                        "optimizer", ["SGD", "AdamW"]
                    ),
                }
            )

        return params

    def _get_efficientdet_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-6, 1e-2, log=True),
            "loader": trial.suggest_categorical("loader", ["inbuilt"]),
        }

        if self.version == "v1":
            params.update(
                {
                    "weights": trial.suggest_categorical(
                        "weights", ["tf_efficientdet_d0"]
                    ),
                    "epochs": trial.suggest_int("epochs", 10, 30),
                    "img_size": trial.suggest_categorical(
                        "img_size", self.V1_IMG_SIZE_OPTIONS
                    ),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                    "optimizer": trial.suggest_categorical(
                        "optimizer", ["momentum", "adam"]
                    ),
                }
            )

        if self.version == "v2":
            params.update(
                {
                    "weights": trial.suggest_categorical(
                        "weights", ["tf_efficientdet_d5"]
                    ),
                    "epochs": trial.suggest_int("epochs", 20, 60),
                    "img_size": trial.suggest_categorical(
                        "img_size", self.V2_IMG_SIZE_OPTIONS
                    ),
                    "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                    "optimizer": trial.suggest_categorical(
                        "optimizer", ["momentum", "adam"]
                    ),
                }
            )

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

        if self.ensemble_size is None:
            logger.warning(
                "Ensemble size not provided in kwargs; cannot suggest trust weights."
            )
            params["fusion_trust_weights"] = None
        else:
            trust_weights = []
            for i in range(self.ensemble_size):
                weight = trial.suggest_float(f"trust_weight_{i}", 0.0, 1.0)
                trust_weights.append(weight)
            params["fusion_trust_weights"] = trust_weights

        return params

    def get_param(
        self,
        trial: optuna.Trial,
        adapter_name: str,
    ) -> Dict[str, Any]:

        param_pairs = [
            ("yolo", partial(self._get_ultralytics_params, name="yolo")),
            ("rtdetr", partial(self._get_ultralytics_params, name="rtdetr")),
            ("fasterrcnn", self._get_fasterrcnn_params),
            ("efficientdet", self._get_efficientdet_params),
            ("ensemblemodel", self._get_ensemble_model_params),
        ]

        for adapter_prefix, param_func in param_pairs:
            if adapter_name.lower().startswith(adapter_prefix):
                return param_func(trial)

        raise ValueError(f"Unknown adapter name: {adapter_name}")
