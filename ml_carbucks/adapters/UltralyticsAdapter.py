import os
import numpy as np
import torch
from typing_extensions import Literal
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import pickle as pkl
from ultralytics.models.yolo import YOLO
from ultralytics.models.rtdetr import RTDETR

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_METRICS,
    ADAPTER_CHECKPOINT,
    BaseDetectionAdapter,
    ADAPTER_PREDICTION,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import (
    postprocess_prediction_nms,
    weighted_boxes_fusion,
)
from ml_carbucks.utils.conversions import (
    convert_coco_to_yolo,
    convert_coco_to_yolo_with_train_val,
)

logger = setup_logger(__name__)

ULTRALYTICS_OPTIMIZER_OPTIONS = Literal["SGD", "Adam", "AdamW", "auto"]


@dataclass
class UltralyticsAdapter(BaseDetectionAdapter):

    # --- HYPER PARAMETERS ---

    optimizer: ULTRALYTICS_OPTIMIZER_OPTIONS = "AdamW"
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    accumulation_steps: int = 1

    # --- SETUP PARAMETERS ---

    seed: int = 42
    strategy: Literal["nms", "wbf"] = "nms"
    training_save: bool = False
    project_dir: str | Path | None = None
    name: str | None = None
    training_augmentations: bool = True

    # --- MAIN METHODS ---

    def fit(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> "UltralyticsAdapter":
        logger.info("Starting training...")

        img_dir, ann_file = datasets[0]
        if len(datasets) > 1:
            logger.error(
                "Multiple datasets provided for training, but only the first will be used."
            )
            logger.error("Multi-dataset training is not yet supported.")
            raise NotImplementedError("Multi-dataset training is not implemented.")

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

        # NOTE: Saving results will be handled incorrectly (weird) if there is no val=True,
        # this is beacuse validation will be skipped and thus no results logged.
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
            accumulate=self.accumulation_steps,
            **extra_params,
        )

        return self

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> ADAPTER_METRICS:
        logger.info("Starting evaluation...")

        img_dir, ann_file = datasets[0]
        if len(datasets) > 1:
            logger.error(
                "Multiple datasets provided for evaluation, but only the first will be used."
            )
            logger.error("Multi-dataset evaluation is not yet supported.")
            raise NotImplementedError("Multi-dataset evaluation is not implemented.")

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
            "map_50_95": results.results_dict["metrics/mAP50-95(B)"],
            "map_75": -np.inf,  # FIXME: verify that the key is correct: results.results_dict["metrics/mAP75(B)"]
            "classes": [],  # FIXME: needs to be added
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
        # NOTE: fit function could be copied here with modifications to log per-epoch results
        if len(train_datasets) > 1 or len(val_datasets) > 1:
            logger.error(
                "Multiple datasets provided for debugging, but only the first of each will be used."
            )
            logger.error("Multi-dataset debugging is not yet supported.")
            raise NotImplementedError("Multi-dataset debugging is not implemented.")

        img_dir, ann_file = train_datasets[0]
        val_img_dir, val_ann_file = val_datasets[0]

        logger.info("Converting COCO annotations to YOLO format...")
        data_yaml = convert_coco_to_yolo_with_train_val(
            str(img_dir), str(ann_file), str(val_img_dir), str(val_ann_file)
        )
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
            name=results_name,
            project=results_path,
            save=True,
            verbose=self.verbose,
            val=True,
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

        results = self.model.val(
            data=data_yaml,
            verbose=self.verbose,
            project=results_path,
            name=f"{results_name}_val",
        )

        metrics: ADAPTER_METRICS = {
            "map_50": results.results_dict["metrics/mAP50(B)"],
            "map_50_95": results.results_dict["metrics/mAP50-95(B)"],
            "map_75": -np.inf,  # FIXME: verify that the key is correct: results.results_dict["metrics/mAP75(B)"]
            "classes": [],  # FIXME: needs to be added
        }

        return metrics

    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model{suffix}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        weights_path = Path(dir) / f"{prefix}weights{suffix}.pt"
        self.model.save(weights_path)  # type: ignore

        weights = torch.load(weights_path, weights_only=False)

        os.remove(weights_path)

        params = self.get_params(skip=["checkpoint"])

        obj = {
            "class_data": {
                "name": self.__class__.__name__,
                "module": self.__class__.__module__,
                "class": self.__class__,
            },
            "params": params,
            "model": weights,
        }

        pkl.dump(obj, open(save_path, "wb"))

        return save_path

    def _load_from_checkpoint(
        self, checkpoint_path: str | Path | dict, **kwargs
    ) -> None:
        model_class: Optional[Type] = kwargs.get("model_class", None)
        if model_class is None:
            raise ValueError(
                "model_class must be provided as a keyword argument to _load_from_checkpoint."
            )

        if isinstance(checkpoint_path, dict):
            obj: ADAPTER_CHECKPOINT = checkpoint_path  # type: ignore
        else:
            obj: ADAPTER_CHECKPOINT = pkl.load(open(checkpoint_path, "rb"))

        obj_class_name = obj["class_data"]["name"]

        if obj_class_name != self.__class__.__name__:
            raise ValueError(
                f"Pickled adapter class mismatch: expected '{self.__class__.__name__}', got '{obj_class_name}'"
            )

        params = obj["params"]

        logger.warning("Overwriting adapter parameters with loaded pickled parameters.")

        self.set_params(params)

        temp_weights_path = Path(f"temp_UltralyticsAdapter_{model_class.__name__}.pt")

        torch.save(obj["model"], temp_weights_path)
        self.model = model_class(str(temp_weights_path))
        os.remove(temp_weights_path)


@dataclass
class YoloUltralyticsAdapter(UltralyticsAdapter):
    # --- MAIN METHODS ---

    def _setup(self) -> "YoloUltralyticsAdapter":
        if self.weights == "DEFAULT":
            self.weights = "yolo11l.pt"

        if self.checkpoint is not None:
            self._load_from_checkpoint(self.checkpoint, model_class=YOLO)

        else:
            self.model = YOLO(str(self.weights))  # type: ignore

        self.model.to(self.device)

        return self

    def predict(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_detections: int = 10,
    ) -> List[ADAPTER_PREDICTION]:

        # NOTE: we are applying NMS/WBF during postprocessing
        results = self.model.predict(  # type: ignore
            source=images,
            imgsz=self.img_size,
            batch=len(images),
            verbose=False,
            # --- Inference-time thresholds ---
            conf=0.0,
            iou=1.0,
            max_det=max_detections * 3,
        )

        all_detections: List[ADAPTER_PREDICTION] = []
        for result in results:

            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            labels = (
                result.boxes.cls + 1
            )  # Ultralytics class ids are 0-based so we increment by 1

            if self.strategy == "nms":
                prediction = postprocess_prediction_nms(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )
            elif self.strategy == "wbf":

                prediction = weighted_boxes_fusion(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

            all_detections.append(prediction)

        return all_detections


@dataclass
class RtdetrUltralyticsAdapter(UltralyticsAdapter):

    # --- MAIN METHODS ---

    def _setup(self) -> "RtdetrUltralyticsAdapter":
        if self.weights == "DEFAULT":
            self.weights = "rtdetr-l.pt"

        if self.checkpoint is not None:
            self._load_from_checkpoint(self.checkpoint, model_class=RTDETR)

        else:
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
        # NOTE: we are applying NMS/WBF during postprocessing
        results = self.model.predict(  # type: ignore
            source=images,
            imgsz=self.img_size,
            batch=len(images),
            verbose=False,
            # --- Inference-time thresholds ---
            conf=0.0,
            max_det=max_detections * 3,
        )

        all_detections: List[ADAPTER_PREDICTION] = []
        for result in results:

            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            labels = (
                result.boxes.cls + 1
            )  # Ultralytics class ids are 0-based so we increment by 1

            if self.strategy == "nms":
                prediction = postprocess_prediction_nms(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )
            elif self.strategy == "wbf":

                prediction = weighted_boxes_fusion(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

            all_detections.append(prediction)

        return all_detections
