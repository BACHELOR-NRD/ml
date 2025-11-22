from pathlib import Path
from typing import Callable, Literal

import optuna
from optuna import Trial

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.optmization.TrialParamWrapper import TrialParamWrapper
from ml_carbucks.utils.logger import setup_logger


logger = setup_logger(__name__)


def create_objective(
    adapter: BaseDetectionAdapter,
    train_datasets: list[tuple],
    val_datasets: list[tuple],
    results_dir: Path,
    param_wrapper_version: Literal["v1", "v2"],
    plot_with_debug: bool = False,
) -> Callable:

    best_score = -float("inf")

    def objective(trial: Trial) -> float:

        try:
            params = TrialParamWrapper(version=param_wrapper_version).get_param(
                trial, adapter.__class__.__name__
            )

            trial_adapter = adapter.clone()
            trial_adapter = trial_adapter.set_params(params)
            trial_adapter.setup()

            if plot_with_debug:
                try:
                    metrics = trial_adapter.debug(
                        train_datasets=train_datasets,
                        val_datasets=val_datasets,
                        results_path=results_dir,
                        results_name=f"debug_{adapter.__class__.__name__}_trial_{trial.number}",
                    )
                except NotImplementedError:
                    logger.info(
                        f"Debug not implemented for {adapter.__class__.__name__}, proceeding with normal fit/evaluate."
                    )
                    trial_adapter.fit(datasets=train_datasets)
                    metrics = trial_adapter.evaluate(datasets=val_datasets)
            else:
                trial_adapter.fit(datasets=train_datasets)
                metrics = trial_adapter.evaluate(datasets=val_datasets)

            score = metrics["map_50"]

            logger.info(
                f"Trial {trial.number} completed with score: {score}, params: {params}, metrics: {metrics}"
            )

            trial.set_user_attr("params", params)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("adapter_params", trial_adapter.get_params())

            nonlocal best_score
            if score > best_score:
                best_score = score

                _ = trial_adapter.save_weights(
                    dir=results_dir,
                    prefix=f"best_weights_{adapter.__class__.__name__}",
                )
                _ = trial_adapter.save_pickled(
                    dir=results_dir,
                    prefix=f"best_pickled_{adapter.__class__.__name__}",
                )

            _ = trial_adapter.save_weights(
                dir=results_dir,
                prefix=f"last_weights_{adapter.__class__.__name__}",
            )
            _ = trial_adapter.save_pickled(
                dir=results_dir,
                prefix=f"last_pickled_{adapter.__class__.__name__}",
            )
            del trial_adapter
            return score
        except optuna.exceptions.TrialPruned as e:
            logger.error(f"Trial {trial.number} pruned: {e}")
            trial.set_user_attr("error", str(e))
            raise

        except Exception as e:
            logger.error(f"Error in objective: {e}")
            trial.set_user_attr("error", str(e))
            return -0.1

    return objective
