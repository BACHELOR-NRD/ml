import json
import datetime as dt
from pathlib import Path
from functools import partial
from typing import Callable, Optional
import warnings

import optuna
import pandas as pd
from optuna import Trial


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
from ml_carbucks import DATA_DIR, RESULTS_DIR

logger = setup_logger(__name__)

warnings.filterwarnings(
    "ignore",
    message="grid_sampler_2d_backward_cuda does not have a deterministic implementation",
)


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
        "best_params": json.dumps(best_trial.params),
        "best_value": best_trial.value if is_single_objective else best_trial.values[0],
        # "model_path": model_path,
        "classes": json.dumps(adapter.classes),
        "best_trial_number": best_trial.number,
        "study_name": name,
        "adapter": adapter.__class__.__name__,
        "n_trials": len(completed_trials),
    }

    with open(result_dir_path / f"results_{adapter.__class__.__name__}.json", "w") as f:
        json.dump(hyper_results, f)

    return hyper_results


def create_objective(
    adapter: BaseDetectionAdapter,
    train_datasets: list[tuple],
    val_datasets: list[tuple],
    results_dir: Path,
) -> Callable:

    def objective(trial: Trial) -> float:

        try:
            params = TrialParamWrapper().get_param(trial, adapter.__class__.__name__)

            trial_adapter = adapter.clone()
            trial_adapter = trial_adapter.set_params(params)
            trial_adapter.setup()
            trial_adapter.fit(datasets=train_datasets)

            metrics = trial_adapter.evaluate(datasets=val_datasets)

            # NOTE: saving all the models may require too much disk space
            # save_path = trial_adapter.save(
            #     dir=results_dir,
            #     prefix=f"trial_{trial.number}_{adapter.__class__.__name__}",
            # )
            score = metrics["map_50_95"]

            logger.info(
                f"Trial {trial.number} completed with score: {score}, params: {params}, metrics: {metrics}"
            )

            trial.set_user_attr("params", params)
            trial.set_user_attr("metrics", metrics)

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
    train_datasets: list[tuple],
    val_datasets: list[tuple],
    n_trials: int = 25,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
    optimization_timeout: Optional[float] = None,
):
    results = []
    for adapter in adapter_list:
        result = execute_simple_study(
            name=runtime,
            results_dir=results_dir,
            n_trials=n_trials,
            create_objective_func=partial(
                create_objective,
                train_datasets=train_datasets,
                val_datasets=val_datasets,
                results_dir=results_dir / "optuna" / f"hyper_{runtime}" / "checkpoints",
            ),
            adapter=adapter,
            patience=patience,
            min_percentage_improvement=min_percentage_improvement,
            optimization_timeout=optimization_timeout,
        )
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "optuna" / f"aggregated_hyper_{runtime}.csv", index=False)


def debug():
    train_datasets = [
        (
            DATA_DIR / "car_dd_testing" / "images" / "train",
            DATA_DIR / "car_dd_testing" / "instances_train_curated.json",
        )
    ]
    val_datasets = [
        (
            DATA_DIR / "car_dd_testing" / "images" / "val",
            DATA_DIR / "car_dd_testing" / "instances_val_curated.json",
        )
    ]
    model = FasterRcnnAdapter(classes=classes)
    model.set_params(
        {
            "img_size": 640,
            "batch_size": 16,
            # "epochs": 1,
            # "epochs": 30,
            # "lr_backbone": 1.1418701765125965e-05,
            # "lr_head": 0.009459099255432072,
            # "weight_decay_backbone": 4.089517605333322e-06,
            # "weight_decay_head": 0.0002135315334108973,
        }
    )
    model.setup()

    model.fit(datasets=train_datasets)  # type: ignore
    metrics = model.evaluate(datasets=val_datasets)  # type: ignore
    print("Metrics:", metrics)


if __name__ == "__main__":
    classes = ["scratch", "dent", "crack"]
    main(
        adapter_list=[
            RtdetrUltralyticsAdapter(classes=classes),
            EfficientDetAdapter(classes=classes),
            YoloUltralyticsAdapter(classes=classes),
            FasterRcnnAdapter(classes=classes),
        ],
        runtime=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        train_datasets=[
            (
                DATA_DIR / "car_dd_testing" / "images" / "train",
                DATA_DIR / "car_dd_testing" / "instances_train_curated.json",
            )
        ],
        val_datasets=[
            (
                DATA_DIR / "car_dd_testing" / "images" / "val",
                DATA_DIR / "car_dd_testing" / "instances_val_curated.json",
            )
        ],
        results_dir=RESULTS_DIR,
        n_trials=20,
        patience=10,
        min_percentage_improvement=0.005,
        optimization_timeout=5 * 3600,  # N hours
    )
    # debug()
