from functools import partial
from pathlib import Path
from typing import Callable, Optional
import datetime as dt

import optuna
from optuna import Trial

import pickle as pkl
from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.adapters.UltralyticsAdapter import (  # noqa: F401
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter  # noqa: F401
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter  # noqa: F401
from ml_carbucks.optmization.EarlyStoppingCallback import create_early_stopping_callback
from ml_carbucks.optmization.TrialParamWrapper import TrialParamWrapper
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks import RESULTS_DIR, DATA_CAR_DD_DIR

logger = setup_logger(__name__)


def execute_simple_study(
    name: str,
    results_dir: Path,
    n_trials: int,
    create_objective_func: Callable,
    adapter: BaseDetectionAdapter,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
    optimization_timeout: Optional[float] = None,
):
    result_dir_path = results_dir / "optuna" / f"hyper_{name}"
    result_dir_path.mkdir(parents=True, exist_ok=True)

    sql_path = (
        results_dir / "optuna" / f"study_{name}" / f"{adapter.__class__.__name__}.db"
    )
    sql_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        load_if_exists=True,
        storage=f"sqlite:///{sql_path}",
    )

    early_callback = None
    if patience > 0:
        early_callback = create_early_stopping_callback(
            name=name,
            study=study,
            patience=patience,
            min_percentage_improvement=min_percentage_improvement,
        )

    study.optimize(
        func=create_objective_func(adapter),
        n_trials=n_trials,
        callbacks=[early_callback] if early_callback else None,
        timeout=optimization_timeout,
        gc_after_trial=True,
    )

    best_trial = None
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed_trials) == 0:
        logger.error(f"No completed trials found for model combination {name}")

    is_single_objective = study.directions is None or len(study.directions) == 1
    if not is_single_objective:
        # NOTE: ROBUST
        best_trial = max(study.best_trials, key=lambda t: min(t.values))
    else:
        best_trial = study.best_trial

    # NOTE: the best model would have to be retrained so that it could be saved properly
    # model_path = adapter.save(result_dir_path)
    hyper_results = {
        "params": best_trial.params,
        "value": best_trial.value if is_single_objective else best_trial.values[0],
        # "model_path": model_path,
        "study_name": name,
        "adapter": adapter.__class__.__name__,
        "n_trials": len(completed_trials),
    }

    with open(result_dir_path / f"{adapter.__class__.__name__}.pkl", "wb") as f:
        pkl.dump(hyper_results, f)


def create_objective(adapter: BaseDetectionAdapter, results_dir: Path) -> Callable:

    def objective(trial: Trial) -> float:

        try:
            params = TrialParamWrapper().get_param(trial, adapter.__class__.__name__)

            trial_adapter = adapter.clone()
            trial_adapter = trial_adapter.set_params(params)
            trial_adapter.setup()
            trial_adapter.fit()

            metrics = trial_adapter.evaluate()

            trial_adapter.save(
                dir=results_dir,
                prefix=f"trial_{trial.number}_{adapter.__class__.__name__}",
            )

            trial.set_user_attr("params", params)
            trial.set_user_attr("metrics", metrics)

            score = metrics["map_50_95"]
            return score
        except optuna.exceptions.TrialPruned as e:
            logger.error("Trial pruned")  # NOTE: this should be replace to logger
            trial.set_user_attr("error", str(e))

            raise e
        except Exception as e:
            logger.error(f"Error in objective: {e}")
            trial.set_user_attr("error", str(e))
            raise e
        finally:
            del trial_adapter  # type: ignore

    return objective


def main(
    adapter_list: list[BaseDetectionAdapter],
    runtime: str,
    results_dir: Path,
    n_trials: int = 25,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
    optimization_timeout: Optional[float] = None,
):

    for adapter in adapter_list:
        execute_simple_study(
            name=runtime,
            results_dir=results_dir,
            n_trials=n_trials,
            create_objective_func=partial(create_objective, results_dir=results_dir),
            adapter=adapter,
            patience=patience,
            min_percentage_improvement=min_percentage_improvement,
            optimization_timeout=optimization_timeout,
        )


if __name__ == "__main__":
    classes = ["scratch", "dent", "crack"]
    main(
        adapter_list=[
            # YoloUltralyticsAdapter(
            #     classes=classes,
            #     metadata={
            #         "data_yaml": "/home/bachelor/ml-carbucks/data/car_dd/dataset.yaml",
            #         "weights": "yolo11l.pt",
            #     },
            # ),
            RtdetrUltralyticsAdapter(
                classes=classes,
                metadata={
                    "data_yaml": "/home/bachelor/ml-carbucks/data/car_dd/dataset.yaml",
                    "weights": "rtdetr-l.pt",
                },
            ),
            EfficientDetAdapter(
                classes=classes,
                metadata={
                    "version": "efficientdet_d0",
                    "train_img_dir": DATA_CAR_DD_DIR / "images" / "train",
                    "train_ann_file": DATA_CAR_DD_DIR / "instances_train_curated.json",
                    "val_img_dir": DATA_CAR_DD_DIR / "images" / "val",
                    "val_ann_file": DATA_CAR_DD_DIR / "instances_val_curated.json",
                },
            ),
            FasterRcnnAdapter(
                classes=classes,
                metadata={
                    "train_img_dir": DATA_CAR_DD_DIR / "images" / "train",
                    "train_ann_file": DATA_CAR_DD_DIR / "instances_train.json",
                    "val_img_dir": DATA_CAR_DD_DIR / "images" / "val",
                    "val_ann_file": DATA_CAR_DD_DIR / "instances_val.json",
                },
            ),
        ],
        runtime=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        results_dir=RESULTS_DIR,
        n_trials=50,
        patience=15,
        min_percentage_improvement=0.005,
        optimization_timeout=6 * 3600,  # 6 hours
    )
