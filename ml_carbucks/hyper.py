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
from ml_carbucks.optmization.execution import execute_simple_study
from ml_carbucks.optmization.hyper_objective import create_objective
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks import OPTUNA_DIR
from ml_carbucks.utils.DatasetsPathManager import DatasetsPathManager
from ml_carbucks.utils.cross_validation import stratified_cross_valitation  # noqa: F401
from ml_carbucks.utils.optimization import get_runtime  # noqa: F401

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
    param_wrapper_version: Literal["h1", "h2", "h3"],
    plot_with_debug: bool = False,
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
            k: v
            for k, v in params.items()
            if k
            not in {"img_size", "epochs", "weights", "batch_size", "accumulation_steps"}
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
                plot_with_debug=plot_with_debug,
            ),
            patience=patience,
            min_percentage_improvement=min_percentage_improvement,
            optimization_timeout=optimization_timeout,
            metadata={
                "runtime": runtime,
                "train_datasets": [(str(ds[0]), str(ds[1])) for ds in train_datasets],
                "val_datasets": [(str(ds[0]), str(ds[1])) for ds in val_datasets],
                "adapter": adapter.__class__.__name__,
                "plot_with_debug": plot_with_debug,
                "param_wrapper_version": param_wrapper_version,
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

    runtime = get_runtime(title="bigger_long")

    adapter_list: list[BaseDetectionAdapter] = [
        YoloUltralyticsAdapter(verbose=True),
        EfficientDetAdapter(verbose=True),
        RtdetrUltralyticsAdapter(verbose=True),
        FasterRcnnAdapter(
            verbose=True, augmentation_noise=False, augmentation_flip=False
        ),
    ]

    # NOTE: defult params are setup here
    main(
        adapter_list=adapter_list,
        runtime=runtime,
        train_datasets=DatasetsPathManager.CARBUCKS_TRAIN_STANDARD,
        val_datasets=DatasetsPathManager.CARBUCKS_VAL_STANDARD,
        param_wrapper_version="h2",  # NOTE: h2 will use bigger image sizes and epochs so it takes longer, h3 is a compromise
        plot_with_debug=True,
        results_dir=OPTUNA_DIR,
        n_trials=30,
        patience=15,
        min_percentage_improvement=0.01,
        optimization_timeout=48 * 3600,  # N hours
    )
