from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.optmization.TrialParamWrapper import TrialParamWrapper
from ml_carbucks.optmization.hyper import logger


import optuna
from optuna import Trial


from pathlib import Path
from typing import Callable


def create_objective(
    adapter: BaseDetectionAdapter,
    train_datasets: list[tuple],
    val_datasets: list[tuple],
    results_dir: Path,
) -> Callable:

    best_score = -float("inf")

    def objective(trial: Trial) -> float:

        try:
            params = TrialParamWrapper().get_param(trial, adapter.__class__.__name__)

            trial_adapter = adapter.clone()
            trial_adapter = trial_adapter.set_params(params)
            trial_adapter.setup()
            trial_adapter.fit(datasets=train_datasets)

            metrics = trial_adapter.evaluate(datasets=val_datasets)

            score = metrics["map_50_95"]

            logger.info(
                f"Trial {trial.number} completed with score: {score}, params: {params}, metrics: {metrics}"
            )

            trial.set_user_attr("params", params)
            trial.set_user_attr("metrics", metrics)

            if score > best_score:
                nonlocal best_score
                best_score = score

                _ = trial_adapter.save(
                    dir=results_dir,
                    prefix=f"best_{adapter.__class__.__name__}",
                )

            _ = trial_adapter.save(
                dir=results_dir,
                prefix=f"last_{adapter.__class__.__name__}",
            )

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
