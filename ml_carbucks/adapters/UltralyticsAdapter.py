import numpy as np
from typing_extensions import Literal
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from ultralytics.models.yolo import YOLO
from ultralytics.models.rtdetr import RTDETR

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_METRICS,
    BaseDetectionAdapter,
    ADAPTER_PREDICTION,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import postprocess_prediction_nms
from ml_carbucks.utils.conversions import convert_coco_to_yolo

logger = setup_logger(__name__)


@dataclass
class UltralyticsAdapter(BaseDetectionAdapter):

    optimizer: Literal["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"] = (
        "auto"
    )
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4

    seed: int = 42
    training_save: bool = True
    verbose: bool = False
    project_dir: str | Path | None = None
    name: str | None = None

    training_augmentations: bool = True

    def fit(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> "UltralyticsAdapter":
        logger.info("Starting training...")

        img_dir, ann_file = datasets[0]
        if len(datasets) > 1:
            logger.warning(
                "Multiple datasets provided for training, but only the first will be used."
            )
            logger.warning("Multi-dataset training is not yet supported.")

        logger.info("Converting COCO annotations to YOLO format...")
        data_yaml = convert_coco_to_yolo(str(img_dir), str(ann_file))
        logger.info(f"YOLO dataset YAML created at: {data_yaml}")

        extra_params = dict()
        if not self.training_augmentations:
            logger.warning(
                "Data augmentations are disabled. This may worsen model performance. It should only be used for debugging purposes."
            )
            extra_params.update(
                {
                    "hsv_h": 0.0,
                    "hsv_s": 0.0,
                    "hsv_v": 0.0,
                    "translate": 0.0,
                    "scale": 0.0,
                    "shear": 0.0,
                    "perspective": 0.0,
                    "flipud": False,
                    "fliplr": 0.0,
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "erasing": 0.0,
                    "auto_augment": None,
                    "augment": False,
                }
            )

        self.model.train(  # type: ignore
            # --- Core parameters ---
            data=data_yaml,
            seed=self.seed,
            name=self.name,
            project=self.project_dir,
            save=self.training_save,
            verbose=self.verbose,
            val=False,
            # --- Hyperparameters ---
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            lr0=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            optimizer=self.optimizer,
            **extra_params,
        )

        return self

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> ADAPTER_METRICS:
        logger.info("Starting evaluation...")

        img_dir, ann_file = datasets[0]
        if len(datasets) > 1:
            logger.warning(
                "Multiple datasets provided for evaluation, but only the first will be used."
            )
            logger.warning("Multi-dataset evaluation is not yet supported.")

        logger.info("Converting COCO annotations to YOLO format...")
        data_yaml = convert_coco_to_yolo(str(img_dir), str(ann_file))
        logger.info(f"YOLO dataset YAML created at: {data_yaml}")

        results = self.model.val(
            data=data_yaml,
            verbose=self.verbose,
            project=self.project_dir,
            name=self.name,
        )

        metrics: ADAPTER_METRICS = {
            "map_50": results.results_dict["metrics/mAP50(B)"],
            "map_75": -np.inf,  # FIXME: verify that the key is correct: results.results_dict["metrics/mAP75(B)"]
            "map_50_95": results.results_dict["metrics/mAP50-95(B)"],
            "classes": [
                i for i in range(1, len(self.classes) + 1)
            ],  # FIXME: try to get class-wise metrics from Ultralytics
        }

        return metrics

    def debug(
        self,
        train_datasets: List[Tuple[str | Path, str | Path]],
        val_datasets: List[Tuple[str | Path, str | Path]],
        results_path: str | Path,
        results_name: str,
        visualize: Literal["every", "last", "none"] = "none",
    ) -> ADAPTER_METRICS:
        logger.error("Debugging not implemented for UltralyticsAdapter.")
        raise NotImplementedError("Debug method is not implemented.")

    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model{suffix}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)  # type: ignore
        return save_path


@dataclass
class YoloUltralyticsAdapter(UltralyticsAdapter):

    weights: str | Path = "yolo11l.pt"

    def setup(self) -> "YoloUltralyticsAdapter":
        self.model = YOLO(str(self.weights))
        self.model.to(self.device)

        return self

    def predict(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_detections: int = 10,
    ) -> List[ADAPTER_PREDICTION]:

        results = self.model.predict(  # type: ignore
            source=images,
            imgsz=self.img_size,
            batch=len(images),
            verbose=False,
            # --- Inference-time thresholds ---
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
        )

        all_detections: List[ADAPTER_PREDICTION] = []
        for result in results:
            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            labels = (
                result.boxes.cls + 1
            )  # Ultralytics class ids are 0-based so we increment by 1

            prediction: ADAPTER_PREDICTION = {
                "boxes": boxes.cpu(),
                "scores": scores.cpu(),
                "labels": labels.cpu().long(),
            }

            all_detections.append(prediction)

        return all_detections


@dataclass
class RtdetrUltralyticsAdapter(UltralyticsAdapter):

    weights: str | Path = "rtdetr-l.pt"

    def setup(self) -> "RtdetrUltralyticsAdapter":
        self.model = RTDETR(str(self.weights))
        self.model.to(self.device)

        return self

    def predict(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_detections: int = 10,
    ) -> List[ADAPTER_PREDICTION]:

        # NOTE: RTDETR does NOT support Non-Maximum Suppression (NMS) at inference time.
        # Therefore, we apply NMS during postprocessing.
        results = self.model.predict(  # type: ignore
            source=images,
            imgsz=self.img_size,
            batch=len(images),
            verbose=False,
            # --- Inference-time thresholds ---
            conf=conf_threshold,
            max_det=max_detections,
        )

        all_detections: List[ADAPTER_PREDICTION] = []
        for result in results:

            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            labels = (
                result.boxes.cls + 1
            )  # Ultralytics class ids are 0-based so we increment by 1

            prediction = postprocess_prediction_nms(
                boxes=boxes,
                scores=scores,
                labels=labels,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )

            all_detections.append(prediction)

        return all_detections
