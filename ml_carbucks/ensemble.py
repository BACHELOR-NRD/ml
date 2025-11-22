import datetime as dt
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pickle as pkl

from ml_carbucks import DATA_DIR, OPTUNA_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_DATASETS,
    BaseDetectionAdapter,
)
from ml_carbucks.adapters.UltralyticsAdapter import (  # noqa: F401
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter  # noqa: F401
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter
from ml_carbucks.optmization.ensemble_objective import create_objective
from ml_carbucks.ensemble.EnsembleModel import EnsembleModel
from ml_carbucks.optmization.ensemble_objective import create_ensembling_opt_prestep
from ml_carbucks.optmization.simple_study import execute_simple_study
from ml_carbucks.utils.ensemble import (
    ScoreDistribution,
)
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_ensemble(
    adapters: List[BaseDetectionAdapter],
    params: Dict[str, Any],
    runtime: str,
    distributions: List[ScoreDistribution],
    final_datasets: ADAPTER_DATASETS | None = None,
) -> EnsembleModel:
    ensemble_adapters = [adapter.clone() for adapter in adapters]
    if final_datasets is None:
        logger.warning(
            "Full datasets for ensemble training not provided. EnsembleModel will be created without fitting."
        )
    else:
        for i in range(len(ensemble_adapters)):
            ensemble_adapters[i] = (
                ensemble_adapters[i].clone().setup().fit(datasets=final_datasets)
            )

    ensemble = EnsembleModel(
        **params,
        adapters=ensemble_adapters,
        distributions=distributions,
    )

    pkl.dump(
        ensemble,
        open(OPTUNA_DIR / "ensembling" / f"ensemble_model_{runtime}.pkl", "wb"),
    )
    return ensemble


def main(
    adapters: list[BaseDetectionAdapter],
    runtime: str,
    results_dir: Path,
    train_folds: list[ADAPTER_DATASETS],
    val_folds: list[ADAPTER_DATASETS],
    final_datasets: ADAPTER_DATASETS | None = None,
    n_trials: int = 25,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
) -> tuple[EnsembleModel, pd.Series]:

    adapters_predictions, ground_truths, distributions = create_ensembling_opt_prestep(
        adapters=adapters,
        train_folds=train_folds,
        val_folds=val_folds,
        runtime=runtime,
        results_dir=results_dir,
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
        ),
        patience=patience,
        min_percentage_improvement=min_percentage_improvement,
        metadata={
            "runtime": runtime,
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
        },
        hyper_suffix="ensemble",
        # append_trials=[default_adapter_params], # NOTE: this could be added to add default ensenble params
    )

    ensemble = create_ensemble(
        adapters=adapters,
        params=result["best_params"],
        runtime=runtime,
        distributions=distributions,
        final_datasets=final_datasets,
    )

    sr = pd.Series(result)
    aggregated_results_path = results_dir / f"ensemble_{runtime}.csv"
    sr.to_csv(aggregated_results_path)
    logger.info(f"Ensemble optimization results saved to {aggregated_results_path}")
    return ensemble, sr


if __name__ == "__main__":
    adapters = [
        # NOTE: paths are placeholders, replace with actual paths
        YoloUltralyticsAdapter.load_pickled("path1"),
        RtdetrUltralyticsAdapter.load_pickled("path2"),
        FasterRcnnAdapter.load_pickled("path3"),
        EfficientDetAdapter.load_pickled("path4"),
    ]
    runtime = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_folds: list[ADAPTER_DATASETS] = []
    validation_folds: list[ADAPTER_DATASETS] = []

    full_datasets: ADAPTER_DATASETS = [
        (
            DATA_DIR / "final_carbucks" / "all" / "images" / "all",
            DATA_DIR / "final_carbucks" / "all" / "instances_all_curated.json",
        )
    ]
    # NOTE: this is to train the ensemble and still have the chance to test it out on unseen data
    # Model should be fully trained at the end
    standard_full: ADAPTER_DATASETS = [
        (
            DATA_DIR / "final_carbucks" / "standard" / "images" / "train",
            DATA_DIR / "final_carbucks" / "standard" / "instances_train_curated.json",
        ),
        (
            DATA_DIR / "final_carbucks" / "standard" / "images" / "val",
            DATA_DIR / "final_carbucks" / "standard" / "instances_val_curated.json",
        ),
    ]
    runtime = runtime
    main(
        adapters=adapters,
        runtime=runtime,
        results_dir=OPTUNA_DIR,
        n_trials=300,
        patience=50,
        min_percentage_improvement=0.01,
        train_folds=training_folds,
        val_folds=validation_folds,
        final_datasets=standard_full,
    )
