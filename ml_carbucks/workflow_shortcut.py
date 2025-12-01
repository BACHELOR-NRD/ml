import optuna

from ml_carbucks import OPTUNA_DIR
from ml_carbucks.adapters import (
    BaseDetectionAdapter,
    YoloUltralyticsAdapter,
    FasterRcnnAdapter,
    EfficientDetAdapter,
    RtdetrUltralyticsAdapter,
)  # noqa: F401
from ml_carbucks.utils.DatasetsPathManager import DatasetsPathManager
from ml_carbucks.custom import main as main_custom
from ml_carbucks.ensemble import main as main_ensemble
from ml_carbucks.utils.optimization import get_runtime

if __name__ == "__main__":
    adapter_list: list[BaseDetectionAdapter] = [
        YoloUltralyticsAdapter(
            **{
                "img_size": 1024,
                "epochs": 40,
                "weights": "yolo11x.pt",
                "checkpoint": None,
                "verbose": True,
                "label_mapper": None,
                "optimizer": "Adam",
                "lr": 0.0001219948413204914,
                "momentum": 0.9346757287744887,
                "weight_decay": 0.0010688292036922472,
                "batch_size": 4,
                "accumulation_steps": 16,
                "scheduler": None,
                "seed": 42,
                "strategy": "nms",
                "training_save": False,
                "project_dir": None,
                "name": None,
                "training_augmentations": True,
            }
        ),
        RtdetrUltralyticsAdapter(
            **{
                "img_size": 1024,
                "epochs": 40,
                "weights": "rtdetr-x.pt",
                "checkpoint": None,
                "verbose": True,
                "label_mapper": None,
                "optimizer": "AdamW",
                "lr": 0.0003025706451014214,
                "momentum": 0.3848287231051003,
                "weight_decay": 0.007029213707652018,
                "batch_size": 4,
                "accumulation_steps": 16,
                "scheduler": None,
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
                "epochs": 20,
                "weights": "V2",
                "checkpoint": None,
                "verbose": True,
                "label_mapper": None,
                "lr_head": 0.0002029384656431124,
                "weight_decay_head": 3.396978549701866e-05,
                "optimizer": "AdamW",
                "clip_gradients": None,
                "momentum": 0.9,
                "strategy": "nms",
                "batch_size": 8,
                "accumulation_steps": 4,
                "scheduler": None,
                "n_classes": 3,
                "training_augmentations": True,
            }
        ),
        EfficientDetAdapter(
            **{
                "img_size": 1024,
                "epochs": 20,
                "weights": "tf_efficientdet_d3",
                "checkpoint": None,
                "verbose": True,
                "label_mapper": None,
                "optimizer": "momentum",
                "lr": 0.0069,
                "weight_decay": 9e-05,
                "loader": "inbuilt",
                "strategy": "nms",
                "batch_size": 4,
                "accumulation_steps": 4,
                "scheduler": None,
                "training_augmentations": True,
                "n_classes": 3,
            }
        ),
    ]

    main_custom(
        adapter_list=[a.clone() for a in adapter_list],
        train_datasets=DatasetsPathManager.CARBUCKS_TRAIN_STANDARD,
        val_datasets=DatasetsPathManager.CARBUCKS_VAL_STANDARD,
    )

    runtime = get_runtime(
        title="finalizing",
    )
    main_ensemble(
        adapters=[a.clone() for a in adapter_list],
        runtime=runtime,
        n_trials=400,
        patience=150,
        results_dir=OPTUNA_DIR,
        param_wrapper_version="e3",
        min_percentage_improvement=0.01,
        sampler=optuna.samplers.TPESampler(n_startup_trials=60),
        train_folds=DatasetsPathManager.CARBUCKS_TRAIN_CV[:2],
        val_folds=DatasetsPathManager.CARBUCKS_VAL_CV[:2],
        final_datasets=DatasetsPathManager.CARBUCKS_TRAIN_ALL,
        skip_trainings=False,
    )
