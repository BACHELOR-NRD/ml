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


def get_trial_params(trial: Trial) -> Dict[str, Any]:
    epochs = trial.suggest_int("epochs", 30, 150)
    batch = trial.suggest_categorical("batch", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.3, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    patience = trial.suggest_int("patience", 25, 75)

    # imgsz = trial.suggest_categorical("imgsz", [320, 640, 960])
    opt = trial.suggest_categorical("optimizer", ["AdamW", "NAdam"])

    return {
        "imgsz": 320,
        "optimizer": opt,
        "epochs": epochs,
        "batch": batch,
        "lr0": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "patience": patience,
    }


def get_trial_params_augumentation(trial: Trial) -> Dict[str, Any]:
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
        "imgsz": 320,
        "epochs": 75,
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
) -> Callable:

    def objective(trial: Trial) -> float:
        model = YOLO(version)
        try:
            if "augumentation" in name.lower():
                params = get_trial_params_augumentation(trial)
            else:
                raise Exception(
                    "Implemented but for this specific run I just want to be sure that augumentation will run"
                )
                params = get_trial_params(trial)

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
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )


# NOTE: This is how to execute hyperparameter optimization, but it takes a lot of time, so I commented it out for now
# RUN_NAME = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_v2"
RUN_NAME = "20250930_augumentation_parameters"
execute_study(name=f"{RUN_NAME}_optuna", n_trials=200)

# NOTE: to view optuna dashboard in terminal: optuna dashboard sqlite:///{sql_path}
