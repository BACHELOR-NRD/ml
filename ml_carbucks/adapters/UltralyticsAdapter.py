import time
import json
import numpy as np
from typing_extensions import Literal
import yaml
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
        data_yaml = self.coco_to_yolo(str(img_dir), str(ann_file))
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
        data_yaml = self.coco_to_yolo(str(img_dir), str(ann_file))
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

    @staticmethod
    def coco_to_yolo(img_dir: str, ann_file: str) -> Path:
        start_time = time.time()
        ann_path = Path(ann_file)
        img_dir_path = Path(img_dir)
        with open(ann_path, "r") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        annotations = coco["annotations"]
        categories = coco["categories"]

        # === remap class ids to contiguous 0-based indices ===
        id_map = {
            cat["id"]: i
            for i, cat in enumerate(sorted(categories, key=lambda x: x["id"]))
        }
        names = {i: cat["name"] for cat, i in zip(categories, id_map.values())}

        # === prepare output paths ===
        labels_dir = img_dir_path.parent.parent / "labels" / img_dir_path.name
        labels_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = ann_path.parent / f"{ann_path.stem}.yaml"

        # === group annotations by image ===
        img_to_anns = {}
        for ann in annotations:
            img_id = ann["image_id"]
            img_to_anns.setdefault(img_id, []).append(ann)

        # === write YOLO label files ===
        for img_id, anns in img_to_anns.items():
            img_info = images[img_id]
            w, h = img_info["width"], img_info["height"]
            label_path = labels_dir / (Path(img_info["file_name"]).stem + ".txt")

            lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in id_map:
                    continue

                bbox = ann["bbox"]  # [x_min, y_min, width, height]
                x_c = (bbox[0] + bbox[2] / 2) / w
                y_c = (bbox[1] + bbox[3] / 2) / h
                bw = bbox[2] / w
                bh = bbox[3] / h

                lines.append(
                    f"{id_map[cat_id]} {round(x_c, 6)} {round(y_c, 6)} {round(bw, 6)} {round(bh, 6)}"
                )

            with open(label_path, "w") as f:
                f.write("\n".join(lines))

        # === create YAML file ===
        dataset_yaml = {
            "train": str(Path(img_dir).resolve()),
            "val": str(Path(img_dir).resolve()),
            "nc": len(categories),
            "names": names,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(dataset_yaml, f, sort_keys=False)
        end_time = time.time()
        elapsed_seconds = end_time - start_time

        logger.info(
            f"COCO to YOLO conversion completed in {elapsed_seconds:.2f} seconds"
        )
        if elapsed_seconds > 15:
            logger.warning(
                "COCO to YOLO conversion took longer than expected. "
                "Consider optimizing this process for large datasets."
            )

        return yaml_path


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
