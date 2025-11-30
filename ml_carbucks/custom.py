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
    # NOTE: pending running as of (2025-11-30)
    adapter_list: list[BaseDetectionAdapter] = [
        YoloUltralyticsAdapter(
            **{
                "img_size": 1024,
                "batch_size": 8,
                "accumulation_steps": 8,
                "epochs": 55,
                "weights": "yolo11x.pt",
                "checkpoint": None,
                "verbose": True,
                "optimizer": "Adam",
                "lr": 0.0001219948413204914,
                "momentum": 0.9346757287744887,
                "weight_decay": 0.0010688292036922472,
                "seed": 42,
                "strategy": "nms",
                "training_save": False,
                "project_dir": None,
                "name": None,
                "training_augmentations": True,
            }
        ),
        FasterRcnnAdapter(
            **{
                "img_size": 1024,
                "epochs": 55,
                "batch_size": 8,
                "accumulation_steps": 4,
                "weights": "V2",
                "checkpoint": None,
                "verbose": True,
                "lr_head": 0.0002229384656431124,
                "weight_decay_head": 3.396978549701866e-05,
                "optimizer": "AdamW",
                "clip_gradients": None,
                "momentum": 0.9,
                "strategy": "nms",
                "scheduler": None,
                "n_classes": 3,
                "training_augmentations": True,
            },
        ),
        RtdetrUltralyticsAdapter(
            **{
                "img_size": 1024,
                "batch_size": 8,
                "accumulation_steps": 8,
                "epochs": 55,
                "weights": "rtdetr-x.pt",
                "checkpoint": None,
                "verbose": True,
                "optimizer": "AdamW",
                "lr": 0.0003025706451014214,
                "momentum": 0.3848287231051003,
                "weight_decay": 0.007029213707652018,
                "seed": 42,
                "strategy": "nms",
                "training_save": False,
                "project_dir": None,
                "name": None,
                "training_augmentations": True,
            },
        ),
        EfficientDetAdapter(
            **{
                "img_size": 1024,
                "epochs": 55,
                "batch_size": 2,
                "accumulation_steps": 8,
                "weights": "tf_efficientdet_d3",
                "checkpoint": None,
                "verbose": True,
                "optimizer": "momentum",
                "lr": 0.0065,
                "weight_decay": 9e-06,
                "loader": "inbuilt",
                "strategy": "nms",
                "scheduler": None,
                "training_augmentations": True,
                "n_classes": 3,
            }
        ),
    ]

    main(
        adapter_list=adapter_list,
        train_datasets=DatasetsPathManager.CARBUCKS_TRAIN_STANDARD,
        val_datasets=DatasetsPathManager.CARBUCKS_VAL_STANDARD,
    )
