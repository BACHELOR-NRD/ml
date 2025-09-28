import datetime as dt
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


RUN_NAME = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def get_trial_params(trial: Trial) -> Dict[str, Any]:
    epochs = trial.suggest_int("epochs", 10, 75)
    batch = trial.suggest_categorical("batch", [8, 16, 32, 64])
    # imgsz = trial.suggest_categorical("imgsz", [320, 640, 960])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    patience = trial.suggest_int("patience", 25, 50)

    return {
        "epochs": epochs,
        "batch": batch,
        "imgsz": 320,
        "lr0": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "patience": patience,
    }


def create_objective(
    version: str,
    data: Path,
    name: str,
    device: str,
) -> Callable:

    def objective(trial: Trial) -> float:
        model = YOLO(version)
        try:
            params = get_trial_params(trial)
            results = model.train(
                pretrained=True,
                seed=42,
                data=data,
                name=name,
                device=device,
                verbose=False,
                **params,
                save=False,
                project=None,
            )

            trial.set_user_attr("params", params)
            trial.set_user_attr("results", results.results_dict)
            fitness = results.fitness

            del results

            return fitness

        except optuna.exceptions.TrialPruned as e:
            print("Trial pruned")  # NOTE: this should be replace to logger
            raise e
        except Exception as e:
            print(f"Error in objective: {e}")
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
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )


# NOTE: This is how to execute hyperparameter optimization, but it takes a lot of time, so I commented it out for now
execute_study(name=f"{RUN_NAME}_optuna")

# NOTE: to view optuna execute in terminal: optuna dashboard sqlite:///{sql_path}
