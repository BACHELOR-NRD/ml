import datetime as dt  # noqa: F401
from pathlib import Path
from typing import Any, Callable, Dict, Union

from ultralytics import YOLO
from optuna import Trial
import optuna

from ml_carbucks import (
    YOLO_PRETRAINED_11N,
    DATA_CAR_DD_YAML,
    RESULTS_DIR,
)


def get_trial_params(trial: Trial, version: int) -> Dict[str, Any]:
    if version == 1:
        return get_params_hyper(trial)
    elif version == 2:
        return get_params_augmentation(trial)
    else:
        raise ValueError(f"Unsupported version: {version}")


def get_params_hyper(trial: Trial) -> Dict[str, Any]:
    epochs = trial.suggest_int("epochs", 30, 150)
    batch = trial.suggest_categorical("batch", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.3, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    patience = trial.suggest_int("patience", 25, 75)

    # imgsz = trial.suggest_categorical("imgsz", [320, 640, 960])
    opt = trial.suggest_categorical("optimizer", ["AdamW", "NAdam"])

    return {
        "optimizer": opt,
        "epochs": epochs,
        "batch": batch,
        "lr0": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "patience": patience,
    }


def get_params_augmentation(trial: Trial) -> Dict[str, Any]:
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 1.0)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 1.0)
    degrees = trial.suggest_float("degrees", 0.0, 45.0)
    translate = trial.suggest_float("translate", 0.0, 0.5)
    scale = trial.suggest_float("scale", 0.0, 1.0)
    shear = trial.suggest_float("shear", -20.0, 20.0)
    fliplr = trial.suggest_float("fliplr", 0.0, 1.0)
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
    mixup = trial.suggest_float("mixup", 0.0, 1.0)

    return {
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "fliplr": fliplr,
        "mosaic": mosaic,
        "mixup": mixup,
    }


def create_objective(
    version: str,
    data: Path,
    name: str,
    device: str,
    results_dir: Path,
    params_version: int,
    override_params: Dict[str, Any] = {},
) -> Callable:

    def objective(trial: Trial) -> float:
        model = YOLO(version)
        try:
            params = get_trial_params(trial, version=params_version)

            if any(param in params for param in override_params):
                raise ValueError("Override params conflict with trial params")

            params.update(override_params)

            results = model.train(
                pretrained=True,
                seed=42,
                data=data,
                name=name,
                device=device,
                verbose=False,
                project=str(results_dir),
                **params,
            )

            trial.set_user_attr("params", params)
            trial.set_user_attr("results", results.results_dict)
            score = results.fitness

            del results

            return score

        except optuna.exceptions.TrialPruned as e:
            print("Trial pruned")  # NOTE: this should be replace to logger
            trial.set_user_attr("error", str(e))
            raise e
        except Exception as e:
            print(f"Error in objective: {e}")
            trial.set_user_attr("error", str(e))
            raise e
        finally:
            del model

    return objective


def execute_study(
    name: str,
    n_trials: int = 25,
    results_dir: Path = RESULTS_DIR,
    version: Union[Path, str] = YOLO_PRETRAINED_11N,
    data: Path = DATA_CAR_DD_YAML,
    direction: str = "maximize",
    device: str = "0",
    params_version: int = 1,
    override_params: Dict[str, Any] = {},
):

    sql_path = results_dir / f"{name}.db"

    study = optuna.create_study(
        direction=direction,
        study_name=name,
        load_if_exists=True,
        storage=f"sqlite:///{sql_path}",
    )

    study.optimize(
        func=create_objective(
            version=str(version),
            data=data,
            name=name,
            device=device,
            results_dir=results_dir,
            params_version=params_version,
            override_params=override_params,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )


# RUN_NAME = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_v2"
RUN_NAME = "20251001_augmentation_parameters"
override_params = {
    "imgsz": 320,
    "optimizer": "AdamW",
    "epochs": 131,
    "batch": 8,
    "lr0": 0.00029631881419241645,
    "momentum": 0.38243835004885135,
    "weight_decay": 9.16499123351809e-05,
    "patience": 31,
}
execute_study(
    name=f"{RUN_NAME}_optuna",
    n_trials=200,
    params_version=2,
    override_params=override_params,
)

# NOTE: to view optuna dashboard in terminal: optuna dashboard sqlite:///{sql_path}
