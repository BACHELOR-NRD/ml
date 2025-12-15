from dataclasses import dataclass
import gc
from pathlib import Path
from typing_extensions import Literal

import torch

from ml_carbucks import RESULTS_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import (
    BaseDetectionAdapter,
    ADAPTER_METRICS,
)
from ml_carbucks.utils.result_saver import ResultSaver
from ml_carbucks.adapters import (
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
    FasterRcnnAdapter,
    EfficientDetAdapter,
)
from ml_carbucks.utils.DatasetsPathManager import DatasetsPathManager
from ml_carbucks.utils.logger import setup_logger


ANALYSIS_DIR = RESULTS_DIR / "m4half_analysis_rerun"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

BASE_PARAMS = {
    # "epochs": 10,
    "batch_size": 16,
    "accumulation_steps": 4,
    "img_size": 320,
    "verbose": True,
}

METRIC = "map_50"
FOLDS_LIMIT = 3
TRAIN_FOLDS = DatasetsPathManager.CARBUCKS_TRAIN_CV[:FOLDS_LIMIT]
VAL_FOLDS = DatasetsPathManager.CARBUCKS_VAL_CV[:FOLDS_LIMIT]

logger = setup_logger(__name__)


def execute_model(
    model_cls: type[BaseDetectionAdapter],
    params: dict,
    train_fold,
    val_fold,
    results_path: Path,
    results_name: str,
) -> ADAPTER_METRICS:
    model = model_cls(**params)
    results = model.debug(
        train_fold, val_fold, results_path=results_path, results_name=results_name
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return results


def analysis_1_dataset_manipulations():
    analysis_debug_path = ANALYSIS_DIR / "analysis_1_dataset_manipulations"
    saver = ResultSaver(
        path=ANALYSIS_DIR,
        name="analysis_1_dataset_manipulations",
        metadata={**BASE_PARAMS, "folds_limit": FOLDS_LIMIT, "metric_name": METRIC},
        append=True,
    )
    singular_models_cls: list[type[BaseDetectionAdapter]] = [
        YoloUltralyticsAdapter,
        RtdetrUltralyticsAdapter,
        FasterRcnnAdapter,
        EfficientDetAdapter,
    ]
    combined_models_cls: list[type[BaseDetectionAdapter]] = [
        FasterRcnnAdapter,
        EfficientDetAdapter,
    ]
    custom_epochs = [60]
    for custom_epoch in custom_epochs:
        for fold_idx, (train, val) in enumerate(
            zip(TRAIN_FOLDS, VAL_FOLDS, strict=True)
        ):
            fold_cnt = fold_idx + 1
            # fmt: off
            cleaned_train = [(
                str(train[0][0]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_cleaned"),
                str(train[0][1]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_cleaned")
            )]
            balanced_train = [(
                str(train[0][0]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_balanced"),
                str(train[0][1]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_balanced")
            )]
            # fmt: on

            for model_cls in singular_models_cls:

                # saver.save(
                #     model_name=model_cls.__name__,
                #     manipulation="carbucks_standard",
                #     fold=fold_cnt,
                #     epochs=custom_epoch,
                #     value=execute_model(
                #         model_cls,
                #         params={**BASE_PARAMS, "epochs": custom_epoch},
                #         train_fold=train,
                #         val_fold=val,
                #         results_path=analysis_debug_path,
                #         results_name=f"fold_{fold_cnt}_{model_cls.__name__}_carbucks_standard_",
                #     )[METRIC],
                # )

                # saver.save(
                #     model_name=model_cls.__name__,
                #     manipulation="carbucks_cleaned",
                #     fold=fold_cnt,
                #     epochs=custom_epoch,
                #     value=execute_model(
                #         model_cls,
                #         params={**BASE_PARAMS, "epochs": custom_epoch},
                #         train_fold=cleaned_train,
                #         val_fold=val,
                #         results_path=analysis_debug_path,
                #         results_name=f"fold_{fold_cnt}_{model_cls.__name__}_carbucks_cleaned_",
                #     )[METRIC],
                # )

                saver.save(
                    model_name=model_cls.__name__,
                    manipulation="carbucks_balanced",
                    fold=fold_cnt,
                    epochs=custom_epoch,
                    value=execute_model(
                        model_cls,
                        params={**BASE_PARAMS, "epochs": custom_epoch},
                        train_fold=balanced_train,
                        val_fold=val,
                        results_path=analysis_debug_path,
                        results_name=f"fold_{fold_cnt}_{model_cls.__name__}_carbucks_balanced2_",
                    )[METRIC],
                )

            # for model_cls in combined_models_cls:

            #     saver.save(
            #         model_name=model_cls.__name__,
            #         manipulation="cardd_plus_carbucks_standard",
            #         fold=fold_cnt,
            #         epochs=custom_epoch,
            #         value=execute_model(
            #             model_cls,
            #             params={**BASE_PARAMS, "epochs": custom_epoch},
            #             train_fold=[DatasetsPathManager.CARDD_TRAIN[0], train[0]],
            #             val_fold=val,
            #             results_path=analysis_debug_path,
            #             results_name=f"fold_{fold_cnt}_{model_cls.__name__}_cardd_plus_carbucks_standard_",
            #         )[METRIC],
            #     )

            #     saver.save(
            #         model_name=model_cls.__name__,
            #         manipulation="cardd_plus_carbucks_cleaned",
            #         fold=fold_cnt,
            #         epochs=custom_epoch,
            #         value=execute_model(
            #             model_cls,
            #             params={**BASE_PARAMS, "epochs": custom_epoch},
            #             train_fold=[
            #                 DatasetsPathManager.CARDD_TRAIN[0],
            #                 cleaned_train[0],
            #             ],
            #             val_fold=val,
            #             results_path=analysis_debug_path,
            #             results_name=f"fold_{fold_cnt}_{model_cls.__name__}_cardd_plus_carbucks_cleaned_",
            #         )[METRIC],
            #     )

            #     saver.save(
            #         model_name=model_cls.__name__,
            #         manipulation="cardd_plus_carbucks_balanced",
            #         fold=fold_cnt,
            #         epochs=custom_epoch,
            #         value=execute_model(
            #             model_cls,
            #             params={**BASE_PARAMS, "epochs": custom_epoch},
            #             train_fold=[
            #                 DatasetsPathManager.CARDD_TRAIN[0],
            #                 balanced_train[0],
            #             ],
            #             val_fold=val,
            #             results_path=analysis_debug_path,
            #             results_name=f"fold_{fold_cnt}_{model_cls.__name__}_cardd_plus_carbucks_balanced_",
            #         )[METRIC],
            #     )


def analysis_2_augmentation_comparison():
    analysis_debug_path = ANALYSIS_DIR / "analysis_2_augmentation_comparison"
    saver = ResultSaver(
        path=ANALYSIS_DIR,
        name="analysis_2_augmentation_comparison",
        metadata={**BASE_PARAMS, "folds_limit": FOLDS_LIMIT, "metric_name": METRIC},
        append=True,
    )

    @dataclass
    class EfficientDetAdapterCustomLoader(EfficientDetAdapter):
        loader: Literal["inbuilt", "custom"] = "custom"

    models_cls: list[type[BaseDetectionAdapter]] = [
        YoloUltralyticsAdapter,
        RtdetrUltralyticsAdapter,
        FasterRcnnAdapter,
        EfficientDetAdapter,
        EfficientDetAdapterCustomLoader,
    ]

    custom_epochs = [20]
    for custom_epoch in custom_epochs:
        for fold_idx, (train, val) in enumerate(
            zip(TRAIN_FOLDS, VAL_FOLDS, strict=True)
        ):
            fold_cnt = fold_idx + 1
            for model_cls in models_cls:

                saver.save(
                    model_name=model_cls.__name__,
                    augmentation=True,
                    fold=fold_cnt,
                    epochs=custom_epoch,
                    value=execute_model(
                        model_cls,
                        params={
                            **BASE_PARAMS,
                            "training_augmentations": True,
                            "epochs": custom_epoch,
                        },
                        train_fold=train,
                        val_fold=val,
                        results_path=analysis_debug_path,
                        results_name=f"fold_{fold_cnt}_{model_cls.__name__}_aug_",
                    )[METRIC],
                )

                saver.save(
                    model_name=model_cls.__name__,
                    augmentation=False,
                    fold=fold_cnt,
                    epochs=custom_epoch,
                    value=execute_model(
                        model_cls,
                        params={
                            **BASE_PARAMS,
                            "training_augmentations": False,
                            "epochs": custom_epoch,
                        },
                        train_fold=train,
                        val_fold=val,
                        results_path=analysis_debug_path,
                        results_name=f"fold_{fold_cnt}_{model_cls.__name__}_noaug_",
                    )[METRIC],
                )


def analysis_3_augmentation_modes():
    analysis_debug_path = ANALYSIS_DIR / "analysis_3_augmentation_modes"
    saver = ResultSaver(
        path=ANALYSIS_DIR,
        name="analysis_3_augmentation_modes",
        metadata={**BASE_PARAMS, "folds_limit": FOLDS_LIMIT, "metric_name": METRIC},
        append=True,
    )

    ultralytics_empty_augs = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "erasing": 0.0,
    }
    # fmt: off
    models_cls_and_augs : list[tuple[type[BaseDetectionAdapter], str, dict]] = [
        (FasterRcnnAdapter, "affine", {"augmentation_affine": True, "augmentation_flip": False, "augmentation_crop": False, "augmentation_color_jitter": False, "augmentation_noise": False}),
        (FasterRcnnAdapter, "flip", {"augmentation_affine": False, "augmentation_flip": True, "augmentation_crop": False, "augmentation_color_jitter": False, "augmentation_noise": False}),
        (FasterRcnnAdapter, "color_jitter", {"augmentation_affine": False, "augmentation_flip": False, "augmentation_crop": False, "augmentation_color_jitter": True, "augmentation_noise": False}),
        (FasterRcnnAdapter, "noise", {"augmentation_affine": False, "augmentation_flip": False, "augmentation_crop": False, "augmentation_color_jitter": False, "augmentation_noise": True}),
        (YoloUltralyticsAdapter, "affine", {**ultralytics_empty_augs, "degrees": 10.0, "translate": 0.1, "scale": 0.4, "shear": 5.0}),
        (YoloUltralyticsAdapter, "flip", {**ultralytics_empty_augs, "flipud": 0.5, "fliplr": 0.5}),
        (YoloUltralyticsAdapter, "color_jitter", {**ultralytics_empty_augs, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4}),
        (YoloUltralyticsAdapter, "noise", {**ultralytics_empty_augs, "bgr": 0.05, "mixup": 0.5, "cutmix": 0.5, "erasing": 0.1}),
    ]
    # fmt: on
    custom_epochs = [20]
    for custom_epoch in custom_epochs:
        for fold_idx, (train, val) in enumerate(
            zip(TRAIN_FOLDS, VAL_FOLDS, strict=True)
        ):
            fold_cnt = fold_idx + 1
            for model_cls, aug_name, aug_params in models_cls_and_augs:

                saver.save(
                    model_name=model_cls.__name__,
                    augmentation_mode=aug_name,
                    fold=fold_cnt,
                    epochs=custom_epoch,
                    value=execute_model(
                        model_cls,
                        params={**BASE_PARAMS, **aug_params, "epochs": custom_epoch},
                        train_fold=train,
                        val_fold=val,
                        results_path=analysis_debug_path,
                        results_name=f"fold_{fold_cnt}_{model_cls.__name__}_{aug_name}_",
                    )[METRIC],
                )


def analysis_4_one_class_only_dataset():
    analysis_debug_path = ANALYSIS_DIR / "analysis_4_one_class_only_dataset"
    saver = ResultSaver(
        path=ANALYSIS_DIR,
        name="analysis_4_one_class_only_dataset",
        metadata={**BASE_PARAMS, "folds_limit": FOLDS_LIMIT, "metric_name": METRIC},
        append=True,
    )

    models_cls: list[type[BaseDetectionAdapter]] = [
        YoloUltralyticsAdapter,
        RtdetrUltralyticsAdapter,
        FasterRcnnAdapter,
        EfficientDetAdapter,
    ]
    custom_epochs = [20]
    for custom_epoch in custom_epochs:
        for fold_idx, (train, val) in enumerate(
            zip(TRAIN_FOLDS, VAL_FOLDS, strict=True)
        ):
            fold_cnt = fold_idx + 1
            for model_cls in models_cls:

                # fmt: off
                train_one_class = [(
                    str(train[0][0]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_crack"),
                    str(train[0][1]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_crack")
                )]
                val_one_class = [(
                    str(val[0][0]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_crack"),
                    str(val[0][1]).replace(f"fold_{fold_cnt}", f"fold_{fold_cnt}_crack")
                )]
                # fmt: on

                saver.save(
                    model_name=model_cls.__name__,
                    fold=fold_cnt,
                    epochs=custom_epoch,
                    value=execute_model(
                        model_cls,
                        params={**BASE_PARAMS, "epochs": custom_epoch},
                        train_fold=train_one_class,
                        val_fold=val_one_class,
                        results_path=analysis_debug_path,
                        results_name=f"fold_{fold_cnt}_{model_cls.__name__}_one_class_only_",
                    )[METRIC],
                )


def analysis_5_score_distributions():
    pass


def analysis_6_strategies_comparison():
    pass


def main():
    analysis_1_dataset_manipulations()
    analysis_2_augmentation_comparison()
    analysis_3_augmentation_modes()
    analysis_4_one_class_only_dataset()


if __name__ == "__main__":
    main()
