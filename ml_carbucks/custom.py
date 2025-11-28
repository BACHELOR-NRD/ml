from functools import partial
from ml_carbucks import OPTUNA_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.adapters.UltralyticsAdapter import YoloUltralyticsAdapter
from ml_carbucks.optmization.execution import execute_custom_study_trial
from ml_carbucks.optmization.hyper_objective import custom_objective_func
from ml_carbucks.utils.DatasetsPathManager import DatasetsPathManager
from ml_carbucks.utils.logger import setup_logger


logger = setup_logger(__name__)


def main(
    adapter_list: list[BaseDetectionAdapter],
    train_datasets: list[tuple],
    val_datasets: list[tuple],
) -> None:

    if len(adapter_list) == 0:
        raise ValueError("adapter_list must contain at least one adapter.")

    results = []
    hyper_name = "custom_params"
    results_dir = OPTUNA_DIR
    models_dir = results_dir / "hyper" / hyper_name / "checkpoints"
    for adapter in adapter_list:
        logger.info(f"Adapter: {adapter.__class__.__name__}")
        params = adapter.get_params()

        result = execute_custom_study_trial(
            hyper_name=hyper_name,
            study_name=adapter.__class__.__name__,
            results_dir=OPTUNA_DIR,
            objective_func=partial(
                custom_objective_func,
                adapter=adapter,
                train_datasets=train_datasets,
                val_datasets=val_datasets,
                results_dir=models_dir,
            ),
            params=params,
            metadata={
                "train_datasets": [(str(ds[0]), str(ds[1])) for ds in train_datasets],
                "val_datasets": [(str(ds[0]), str(ds[1])) for ds in val_datasets],
                "adapter": adapter.__class__.__name__,
            },
        )

        logger.info(f"Result: {result}")
        results.append(result)

    logger.info("Custom study trials completed.")


if __name__ == "__main__":
    # NOTE: pending running as of (2025-11-28)
    adapter_list: list[BaseDetectionAdapter] = [
        YoloUltralyticsAdapter(
            **{
                "img_size": 768,
                "batch_size": 8,
                "epochs": 36,
                "weights": "yolo11x.pt",
                "checkpoint": None,
                "verbose": True,
                "optimizer": "Adam",
                "lr": 0.0001419948413204914,
                "momentum": 0.9346757287744887,
                "weight_decay": 0.0010688292036922472,
                "accumulation_steps": 4,
                "seed": 42,
                "strategy": "nms",
                "training_save": False,
                "project_dir": None,
                "name": None,
                "training_augmentations": True,
            }
        ),
    ]

    main(
        adapter_list=adapter_list,
        train_datasets=DatasetsPathManager.CARBUCKS_TRAIN_STANDARD,
        val_datasets=DatasetsPathManager.CARBUCKS_VAL_STANDARD,
    )
