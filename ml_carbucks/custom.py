from functools import partial
from ml_carbucks import OPTUNA_DIR
from ml_carbucks.adapters import (  # noqa: F401
    BaseDetectionAdapter,
    YoloUltralyticsAdapter,
    FasterRcnnAdapter,
    EfficientDetAdapter,
    RtdetrUltralyticsAdapter,
)

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
    adapter_list: list[BaseDetectionAdapter] = [
        EfficientDetAdapter(
            **{
                "img_size": 1024,
                "epochs": 30,
                "weights": "tf_efficientdet_d3",
                "checkpoint": None,
                "verbose": True,
                "label_mapper": None,
                "optimizer": "momentum",
                "lr": 0.0030659478747789057,
                "weight_decay": 5.628823667567728e-05,
                "loader": "inbuilt",
                "strategy": "nms",
                "batch_size": 4,
                "accumulation_steps": 16,
                "scheduler": None,
                "training_augmentations": True,
                "n_classes": 3,
            }
        ),
        RtdetrUltralyticsAdapter(
            **{
                "img_size": 512,
                "epochs": 17,
                "weights": "rtdetr-l.pt",
                "checkpoint": None,
                "verbose": True,
                "label_mapper": None,
                "optimizer": "AdamW",
                "lr": 0.0003175834013298623,
                "momentum": 0.8495963023753279,
                "weight_decay": 4.829642412571587e-05,
                "batch_size": 32,
                "accumulation_steps": 2,
                "scheduler": None,
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
