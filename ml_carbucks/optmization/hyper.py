import datetime as dt
from pathlib import Path
from functools import partial
from typing import Optional
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
from ml_carbucks.optmization.objective import create_objective
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks import DATA_DIR, RESULTS_DIR

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
    n_trials: int = 25,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
    optimization_timeout: Optional[float] = None,
):
    results = []
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
                results_dir=results_dir / "optuna" / "hyper" / f"checkpoints_{runtime}",
            ),
            patience=patience,
            min_percentage_improvement=min_percentage_improvement,
            optimization_timeout=optimization_timeout,
            metadata={
                "runtime": runtime,
                "train_datasets": train_datasets,
                "val_datasets": val_datasets,
                "adapter": adapter.__class__.__name__,
            },
            append_trials=[default_adapter_params],
        )
        results.append(result)

    df = pd.DataFrame(results)
    aggregated_results_path = results_dir / "optuna" / f"aggregated_hyper_{runtime}.csv"
    df.to_csv(aggregated_results_path, index=False)
    logger.info(f"Aggregated results saved to {aggregated_results_path}")


if __name__ == "__main__":
    classes = ["scratch", "dent", "crack"]
    runtime = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    main(
        adapter_list=[
            RtdetrUltralyticsAdapter(classes=classes, training_save=False),
            EfficientDetAdapter(classes=classes),
            YoloUltralyticsAdapter(classes=classes, training_save=False),
            FasterRcnnAdapter(classes=classes),
        ],
        runtime=runtime,
        train_datasets=[
            (
                DATA_DIR
                / "combinations"
                / "cardd_plus_carbucks_splitted"
                / "images"
                / "train",
                DATA_DIR
                / "combinations"
                / "cardd_plus_carbucks_splitted"
                / "instances_train_curated.json",
            ),
        ],
        val_datasets=[
            (
                DATA_DIR
                / "combinations"
                / "cardd_plus_carbucks_splitted"
                / "images"
                / "val",
                DATA_DIR
                / "combinations"
                / "cardd_plus_carbucks_splitted"
                / "instances_val_curated.json",
            )
        ],
        results_dir=RESULTS_DIR,
        n_trials=30,
        patience=15,
        min_percentage_improvement=0.02,
        optimization_timeout=8 * 3600,  # N hours
    )
