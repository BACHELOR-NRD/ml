import datetime as dt
from pathlib import Path
from functools import partial
from typing import Literal, Optional
import warnings

import pandas as pd

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.adapters.UltralyticsAdapter import (  # noqa: F401
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter  # noqa: F401
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter  # noqa: F401
from ml_carbucks.optmization.simple_study import execute_simple_study
from ml_carbucks.optmization.hyper_objective import create_objective
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks import DATA_DIR, OPTUNA_DIR
from ml_carbucks.utils.cross_validation import stratified_cross_valitation  # noqa: F401

logger = setup_logger(__name__)

warnings.filterwarnings(
    "ignore",
    message="grid_sampler_2d_backward_cuda does not have a deterministic implementation",
)


def main(
    adapter_list: list[BaseDetectionAdapter],
    runtime: str,
    results_dir: Path,
    train_datasets: list[tuple],
    val_datasets: list[tuple],
    param_wrapper_version: Literal["v1", "v2"],
    n_trials: int = 25,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
    optimization_timeout: Optional[float] = None,
) -> pd.DataFrame:
    if len(adapter_list) == 0:
        raise ValueError("adapter_list must contain at least one adapter.")
    results = []
    models_dir = results_dir / "hyper" / runtime / "checkpoints"
    for adapter in adapter_list:

        # NOTE: this is to see if default params can outperform hyperparams
        # since some parameters are mock values then we exclude them
        params = adapter.get_params()
        default_adapter_params = {
            k: v for k, v in params.items() if k not in {"img_size", "epochs"}
        }

        result = execute_simple_study(
            hyper_name=runtime,
            study_name=adapter.__class__.__name__,
            results_dir=results_dir,
            n_trials=n_trials,
            create_objective_func=partial(
                create_objective,
                adapter=adapter,
                train_datasets=train_datasets,
                val_datasets=val_datasets,
                results_dir=models_dir,
                param_wrapper_version=param_wrapper_version,
            ),
            patience=patience,
            min_percentage_improvement=min_percentage_improvement,
            optimization_timeout=optimization_timeout,
            metadata={
                "runtime": runtime,
                "train_datasets": [(str(ds[0]), str(ds[1])) for ds in train_datasets],
                "val_datasets": [(str(ds[0]), str(ds[1])) for ds in val_datasets],
                "adapter": adapter.__class__.__name__,
            },
            append_trials=[default_adapter_params],
        )

        extended_result = result.copy()
        extended_result["models_dir"] = str(models_dir)

        results.append(extended_result)

        # stratified_cross_valitation(
        #     hyper_results=result,               #Set the annotations_path,dataset_dir and cvfolds if needed
        #     results_dir=results_dir,
        # )

    df = pd.DataFrame(results)
    aggregated_results_path = results_dir / f"aggregated_hyper_{runtime}.csv"
    df.to_csv(aggregated_results_path, index=False)
    logger.info(f"Aggregated results saved to {aggregated_results_path}")
    return df


if __name__ == "__main__":

    runtime = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime += "big_resolution_carbucks"
    train_datasets = [
        (
            DATA_DIR / "final_carbucks" / "standard" / "images" / "train",
            DATA_DIR / "final_carbucks" / "standard" / "instances_train_curated.json",
        ),
    ]
    val_datasets = [
        (
            DATA_DIR / "final_carbucks" / "standard" / "images" / "val",
            DATA_DIR / "final_carbucks" / "standard" / "instances_val_curated.json",
        )
    ]
    results_dir = OPTUNA_DIR

    # NOTE: this runs broad hyperparameter optimization with small image_resolution
    main(
        adapter_list=[
            EfficientDetAdapter(),
            FasterRcnnAdapter(),
            YoloUltralyticsAdapter(),
            RtdetrUltralyticsAdapter(),
        ],
        runtime=runtime,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        param_wrapper_version="v2",  # NOTE: v2 will use bigger image sizes and epochs so it takes longer
        results_dir=OPTUNA_DIR,
        n_trials=1,
        patience=15,
        min_percentage_improvement=0.01,
        optimization_timeout=4 * 3600,  # N hours
    )
