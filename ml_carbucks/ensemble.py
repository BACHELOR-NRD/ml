import datetime as dt  # noqa: F401
from pathlib import Path
from functools import partial
from typing import List, Literal, Type

import optuna
import pandas as pd

from ml_carbucks import OPTUNA_DIR, PRODUCTS_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_DATASETS,
    BaseDetectionAdapter,
)
from ml_carbucks.adapters.UltralyticsAdapter import (  # noqa: F401
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter  # noqa: F401
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter  # noqa: F401
from ml_carbucks.ensemble.EnsembleModel import EnsembleModel
from ml_carbucks.optmization.ensemble_objective import (
    create_ensemble,
    create_objective,
    create_ensembling_opt_prestep,
)
from ml_carbucks.optmization.execution import execute_simple_study
from ml_carbucks.utils.DatasetsPathManager import DatasetsPathManager
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.optimization import get_runtime

logger = setup_logger(__name__)


def load_adapters_from_hyperopt(
    hyper_runtime: str, load_pattern: str = "best_pickled_*_model.pkl"
) -> List[BaseDetectionAdapter]:
    """
    A function that takes care of automatically loading all the adapters from given hyperopt runtime.
    """
    hyperopt_models_dir = OPTUNA_DIR / "hyper" / hyper_runtime / "checkpoints"

    # walk thru all the files inside the directory
    possible_adapter_classes: List[Type[BaseDetectionAdapter]] = [
        YoloUltralyticsAdapter,
        RtdetrUltralyticsAdapter,
        FasterRcnnAdapter,
        EfficientDetAdapter,
    ]

    adapters: List[BaseDetectionAdapter] = []

    for file in hyperopt_models_dir.glob(load_pattern):
        for adapter_class in possible_adapter_classes:
            try:
                adapter = adapter_class(checkpoint=str(file))
                adapters.append(adapter)
                logger.info(f"Loaded adapter from {file}")
                break
            except ValueError:
                pass

    if len(adapters) == 0:
        raise ValueError(
            f"No adapters found in {hyperopt_models_dir} with pattern {load_pattern}"
        )

    return adapters


def main(
    adapters: list[BaseDetectionAdapter],
    runtime: str,
    results_dir: Path,
    train_folds: list[ADAPTER_DATASETS],
    val_folds: list[ADAPTER_DATASETS],
    param_wrapper_version: Literal["e1", "e2", "e3"],
    final_datasets: ADAPTER_DATASETS | None = None,
    n_trials: int = 25,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
    n_jobs: int = 1,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> tuple[EnsembleModel, pd.Series]:

    adapters_predictions, ground_truths, distributions, metadata = (
        create_ensembling_opt_prestep(
            adapters=adapters,
            train_folds=train_folds,
            val_folds=val_folds,
            results_dir=results_dir,
        )
    )

    result = execute_simple_study(
        hyper_name=runtime,
        study_name="ensemble",
        results_dir=results_dir,
        n_trials=n_trials,
        create_objective_func=partial(
            create_objective,
            adapters_predictions=adapters_predictions,
            ground_truths=ground_truths,
            distributions=distributions,
            param_wrapper_version=param_wrapper_version,
        ),
        patience=patience,
        min_percentage_improvement=min_percentage_improvement,
        metadata={
            "runtime": runtime,
            "param_wrapper_version": param_wrapper_version,
            "train_folds": [
                [str(dataset[0]), str(dataset[1])]
                for fold in train_folds
                for dataset in fold
            ],
            "val_folds": [
                [str(dataset[0]), str(dataset[1])]
                for fold in val_folds
                for dataset in fold
            ],
            **metadata,
        },
        study_attributes={
            "param_wrapper_version": param_wrapper_version,
            **metadata,
        },
        n_jobs=n_jobs,
        sampler=sampler,
        hyper_suffix="ensemble",
        # append_trials=[default_adapter_params], # NOTE: this could be added to add default ensenble params
    )

    ensemble = create_ensemble(
        adapters=adapters,
        params=result["best_params"],
        runtime=runtime,
        distributions=distributions,
        final_datasets=final_datasets,
        results_dir=results_dir,
    )

    sr = pd.Series(result)
    aggregated_results_path = results_dir / f"ensemble_{runtime}.csv"
    sr.to_csv(aggregated_results_path)
    logger.info(f"Ensemble optimization results saved to {aggregated_results_path}")
    return ensemble, sr


if __name__ == "__main__":

    runtime = get_runtime(
        title="demo_combined_explorative",
        override="20251201_114723_demo_combined_explorative",
    )

    # adapters = load_adapters_from_hyperopt(
    #     runtime, load_pattern="best_pickled_*_model.pkl"
    # )

    # NOTE: manual override for debugging
    adapters = [
        YoloUltralyticsAdapter(
            checkpoint=PRODUCTS_DIR / "best_pickled_YoloUltralyticsAdapter_model.pkl"
        ),
        RtdetrUltralyticsAdapter(
            checkpoint=PRODUCTS_DIR / "best_pickled_RtdetrUltralyticsAdapter_model.pkl"
        ),
        FasterRcnnAdapter(
            checkpoint=PRODUCTS_DIR / "best_pickled_FasterRcnnAdapter_model.pkl"
        ),
        EfficientDetAdapter(
            checkpoint=PRODUCTS_DIR / "best_pickled_EfficientDetAdapter_model.pkl"
        ),
    ]

    main(
        adapters=adapters,
        runtime=runtime,
        results_dir=OPTUNA_DIR,
        n_trials=400,
        patience=100,
        # NOTE: e2 only allows WBF and e1 only NMS, e3 is combined but also unnecessary exponents for nms
        param_wrapper_version="e3",
        min_percentage_improvement=0.01,
        n_jobs=1,
        # NOTE: default n_startup_trials is 10
        sampler=optuna.samplers.TPESampler(n_startup_trials=60),
        train_folds=[DatasetsPathManager.CARBUCKS_TRAIN_STANDARD],
        val_folds=[DatasetsPathManager.CARBUCKS_VAL_STANDARD],
        final_datasets=DatasetsPathManager.CARBUCKS_TRAIN_ALL,
    )
