import gc
from pathlib import Path
import pickle as pkl  # noqa: F401

import torch  # noqa: F401
from ml_carbucks import OPTUNA_DIR, PRODUCTS_DIR
from ml_carbucks.adapters import (
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
    FasterRcnnAdapter,
    EfficientDetAdapter,
    BaseDetectionAdapter,
)
from ml_carbucks.adapters.EnsembleModel import EnsembleModel
from ml_carbucks.utils.DatasetsPathManager import DatasetsPathManager
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import plot_pr_curves_with_ap50  # noqa: F401
from typing import Any

from ml_carbucks import RESULTS_DIR
from ml_carbucks.utils.result_saver import ResultSaver

logger = setup_logger(__name__)


def evaluate_adapters():

    adapter_classes: list[type[BaseDetectionAdapter]] = [
        YoloUltralyticsAdapter,
        RtdetrUltralyticsAdapter,
        FasterRcnnAdapter,
        EfficientDetAdapter,
    ]

    adapter_list = [
        adapter_class(
            checkpoint=PRODUCTS_DIR / f"best_pickled_{adapter_class.__name__}_model.pkl"
        )
        for adapter_class in adapter_classes
    ]

    results = []

    for adapter in adapter_list:
        res = adapter.evaluate(datasets=DatasetsPathManager.CARBUCKS_VAL_STANDARD)
        results.append(res)

    for adapter, res in zip(adapter_list, results):
        logger.info(
            f"Results for {adapter.__class__.__name__}: {res['map_50'], res['map_per_class']}"
        )

    """
    INFO __main__ 10:35:40 | Results for YoloUltralyticsAdapter: (0.15555486945584648, None)
    INFO __main__ 10:35:40 | Results for RtdetrUltralyticsAdapter: (0.10799950727913095, None)
    INFO __main__ 10:35:40 | Results for FasterRcnnAdapter: (0.1394740492105484, [0.23941896855831146, 0.12206945568323135, 0.05693374201655388])
    INFO __main__ 10:35:40 | Results for EfficientDetAdapter: (0.131548210978508, [0.2427784949541092, 0.13107948005199432, 0.02078666165471077])
    """


def evaluate_ensemble():
    ensemble = EnsembleModel(
        checkpoint=OPTUNA_DIR
        / "ensemble"
        / "20251201_114723_demo_combined_explorative"
        / "ensemble_model20251201_114723_demo_combined_explorative.pkl",
    )

    ensemble.set_params({"verbose": True})

    res = ensemble.evaluate(datasets=DatasetsPathManager.CARBUCKS_VAL_STANDARD)
    logger.info(f"Ensemble results: {res['map_50'], res['map_per_class']}")


def debug_ensemble_checkpoint_cpu():
    checkpoint_path = Path(
        "/home/damian/Desktop/Projects/Bachelor/ml/ensemble_model.pkl"
    )

    # obj = pkl.load(open(checkpoint_path, "rb"))
    ensemlbe = EnsembleModel(checkpoint=checkpoint_path)

    ensemlbe.save(checkpoint_path.parent / "ensemble_model_cpu.pkl", suffix="_cpu")
    print("here")


def augumentation_analysis():

    AUG_METRIC = "map_50"
    AUG_BASE_PARAMS: dict[str, Any] = {
        "batch_size": 8,
        "accumulation_steps": 4,
        "img_size": 320,
    }

    from dataclasses import dataclass
    from typing import Literal

    @dataclass
    class EfficientDetAdapterCustomLoader(EfficientDetAdapter):
        loader: Literal["inbuilt", "custom"] = "custom"

    model_clsasses: list[type[BaseDetectionAdapter]] = [
        YoloUltralyticsAdapter,
        RtdetrUltralyticsAdapter,
        FasterRcnnAdapter,
        EfficientDetAdapter,
        EfficientDetAdapterCustomLoader,
    ]

    saver2 = ResultSaver(
        path=RESULTS_DIR,
        name="saver2_crossval_augmentations_comparisons",
        metadata=AUG_BASE_PARAMS,
    )

    EPOCHS_TO_TEST = [10, 20, 30]

    logger.info("Starting cross-validation for models augmentations comparisons")
    for epoch_count in EPOCHS_TO_TEST:
        for train_idx, (train, val) in enumerate(
            zip(
                DatasetsPathManager.CARBUCKS_TRAIN_CV,
                DatasetsPathManager.CARBUCKS_VAL_CV,
                strict=True,
            )
        ):
            for model_cls in model_clsasses:

                model_aug = model_cls(**AUG_BASE_PARAMS, training_augmentations=True, epochs=epoch_count)  # type: ignore
                model_aug.fit(train)
                res_aug = model_aug.evaluate(val)
                saver2.save(
                    model_name=model_aug.__class__.__name__,
                    augmentation=True,
                    fold=train_idx,
                    metric_name=AUG_METRIC,
                    epochs=epoch_count,
                    metric_value=res_aug[AUG_METRIC],
                )
                del model_aug
                del res_aug
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                model_noaug = model_cls(**AUG_BASE_PARAMS, training_augmentations=False, epochs=epoch_count)  # type: ignore
                model_noaug.fit(train)
                res_noaug = model_noaug.evaluate(val)
                saver2.save(
                    model_name=model_noaug.__class__.__name__,
                    augmentation=False,
                    fold=train_idx,
                    metric_name=AUG_METRIC,
                    epochs=epoch_count,
                    metric_value=res_noaug[AUG_METRIC],
                )
                del model_noaug
                del res_noaug
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    logger.info("Cross-validation for models augmentations comparisons completed.")
    logger.info(saver2.data)


def aug_expo():
    from ultralytics.models.yolo import YOLO
    from ml_carbucks.utils.result_saver import ResultSaver
    from ml_carbucks import RESULTS_DIR

    logger.info(RESULTS_DIR)
    params = {
        "imgsz": 386,
        "epochs": 60,
        "batch": 16,
        "nbs": 64,
    }
    saver = ResultSaver(
        path=RESULTS_DIR,
        name="yolo_augmentation_exploration_final_carbucks",
        metadata={
            **params,
            "dataset": "final_carbucks_combined_standard",
        },
        append=True,
    )
    empty_aug = {
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
    aug_configs = {
        # "empty_aug": {**empty_aug},
        # "flips_only": {**empty_aug, "fliplr": 0.5, "flipud": 0.5},
        # "light_aug": {
        #     **empty_aug,
        #     "hsv_h": 0.015,
        #     "hsv_s": 0.7,
        #     "hsv_v": 0.4,
        #     "fliplr": 0.5,
        # },
        # "rotation_translation": {**empty_aug, "degrees": 10.0, "translate": 0.1},
        "bgr_perspective_translate_scale": {
            **empty_aug,
            "bgr": 0.5,
            "perspective": 0.5,
            "translate": 0.2,
            "scale": 0.2,
        },
        "full_aug": {
            **empty_aug,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.1,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.5,
            "bgr": 0.5,
            "perspective": 0.5,
            "erasing": 0.25,
        },
    }

    for aug_name, aug_params in aug_configs.items():
        logger.info(f"Starting training with augmentation: {aug_name}")

        model = YOLO("yolo11m.pt")
        res = model.train(
            data="/home/damian/Desktop/Projects/Bachelor/ml/data/final_carbucks/standard/dataset_combined.yaml",
            **params,
            project=RESULTS_DIR / "yolo_augmentation_exploration",
            name=aug_name,
            **aug_params,
        )

        map_50 = res.results_dict["metrics/mAP50(B)"]  # type: ignore
        saver.save(
            model_name="YOLO11m",
            augmentation=aug_name,
            metric_name="map_50",
            metric_value=map_50,
        )
        logger.info(f"Completed training with augmentation: {aug_name}: mAP50={map_50}")


def aug_fasterrcnn():
    from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter

    augmentation_names = [
        "augmentation_affine",
        "augmentation_flip",
        "augmentation_crop",
        "augmentation_color_jitter",
        "augmentation_noise",
    ]

    saver = ResultSaver(
        path=RESULTS_DIR,
        name="fasterrcnn_augmentation_exploration_final_carbucks",
        metadata={},
        append=True,
    )

    for i in range(1 << len(augmentation_names)):
        aug_params = {}
        aug_name_parts = []
        for j in range(len(augmentation_names)):
            if (i & (1 << j)) != 0:
                aug_params[augmentation_names[j]] = True
                aug_name_parts.append(
                    augmentation_names[j][13:]
                )  # remove "augmentation_"
            else:
                aug_params[augmentation_names[j]] = False
        aug_name = "_".join(aug_name_parts) if aug_name_parts else "no_augmentation"
        logger.info(f"Starting training with augmentation: {aug_name}")

        model = FasterRcnnAdapter(
            batch_size=16,
            accumulation_steps=2,
            img_size=320,
            epochs=15,
            verbose=True,
            training_augmentations=True,
            **aug_params,
        )
        res = model.debug(
            DatasetsPathManager.CARBUCKS_TRAIN_STANDARD,
            DatasetsPathManager.CARBUCKS_VAL_STANDARD,
            results_path=RESULTS_DIR / "fasterrcnn_augmentation_exploration",
            results_name=aug_name,
        )

        map_50 = res["map_50"]
        saver.save(
            model_name="FasterRCNN",
            augmentation=aug_name,
            metric_name="map_50",
            metric_value=map_50,
            bitmask=i,
            binary_bitmask=format(i, f"0{len(augmentation_names)}b"),
        )

        del model
        del res
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info(f"Completed training with augmentation: {aug_name}: mAP50={map_50}")


if __name__ == "__main__":
    logger.info("Starting main")
    # evaluate_adapters()
    # evaluate_ensemble()
    # debug_ensemble_checkpoint_cpu()
    # augumentation_analysis()
    # aug_expo()
    aug_fasterrcnn()
