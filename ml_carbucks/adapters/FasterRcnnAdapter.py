from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.datasets import CocoDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


class FasterRcnnAdapter(BaseDetectionAdapter):
    def load_model(self):
        logger.info("Loading Faster R-CNN model...")
        # Implementation for loading or creating a Faster R-CNN model goes here

        if self.model_path and self.model_path.exists():
            self._load_existing_model()
        else:
            self._create_model()

        self.model = self.model.to(self.device)

    def _create_model(self):
        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, len(self.classes) + 1  # +1 for background
        )

    def _load_existing_model(self):
        logger.info(f"Loading existing model from {self.model_path}...")

        self.model = fasterrcnn_resnet50_fpn(
            pretrained=False, num_classes=len(self.classes) + 1
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)  # type: ignore
        self.model.load_state_dict(checkpoint)

    def setup(self):
        logger.info("Loading datasets...")
        batch_size = self.hparams["batch_size"]

        train_img_dir = self.datasets.get("train_img_dir", None)
        train_ann_file = self.datasets.get("train_ann_file", None)

        val_img_dir = self.datasets.get("val_img_dir", None)
        val_ann_file = self.datasets.get("val_ann_file", None)

        if (
            not train_img_dir
            or not train_ann_file
            or not val_img_dir
            or not val_ann_file
        ):
            raise ValueError(
                "Both train and val dataset paths must be provided in datasets to load data."
            )

        train_dataset = COCODetectionWrapper(
            img_folder=train_img_dir,
            ann_file=train_ann_file,
            transforms=self._create_transforms(is_training=True),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,  # 2–8 is typical, memory permitting
            shuffle=True,
            num_workers=batch_size // 2,  # adjust based on your CPU
            pin_memory=True,
            collate_fn=collate_fn,  # crucial
        )

        val_dataset = COCODetectionWrapper(
            img_folder=val_img_dir,
            ann_file=val_ann_file,
            transforms=self._create_transforms(is_training=False),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=batch_size // 2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def fit(self):
        logger.info("Starting training...")

        epochs = self.hparams["epochs"]

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.train_epoch()

    def _get_optimizer(self):

        if hasattr(self, "optimizer"):
            return self.optimizer

        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 5e-5, "weight_decay": 1e-5},
                {"params": head_params, "lr": 5e-4, "weight_decay": 1e-4},
            ]
        )
        return self.optimizer

    def train_epoch(self):

        self.model.train()
        total_loss = 0.0

        for imgs, targets in tqdm(self.train_loader):
            imgs = list(img.to(self.device) for img in imgs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            self._get_optimizer().zero_grad()
            loss.backward()  # type: ignore
            self._get_optimizer().step()

            total_loss += loss.item()  # type: ignore

    def evaluate(self) -> Dict[str, float]:
        logger.info("Starting evaluation...")

        self.model.eval()

        metric = MeanAveragePrecision()
        with torch.no_grad():
            for imgs, targets in self.val_loader:
                imgs = list(img.to(self.device) for img in imgs)
                outputs = self.model(imgs)

                # Move targets and outputs to CPU for metric computation
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                outputs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]

                metric.update(outputs_cpu, targets_cpu)

        results = metric.compute()

        metrics = {
            "map_50": results["map_50"].item(),
            "map_50_95": results["map"].item(),
        }

        return metrics

    def predict(self, images: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError("Predict method is not yet implemented.")

    def save_model(self, save_path: Path | str):
        logger.info(f"Saving model to {save_path}...")
        torch.save(self.model.state_dict(), save_path)

    def _create_transforms(self, is_training: bool) -> A.Compose:
        img_size = self.hparams["img_size"]

        base = [
            # Always resize and pad to square input
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,  # constant padding
                fill=(0, 0, 0),
            ),
        ]

        if is_training:
            # --- Spatial augmentations (geometry) ---
            base += [
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # up to ±5% shift
                    scale_limit=0.1,  # up to ±10% zoom
                    rotate_limit=10,  # ±10 degrees
                    border_mode=0,
                    fill=(0, 0, 0),
                    p=0.7,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, p=0.3),
                # --- Photometric augmentations (color) ---
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.3, p=1
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=15,
                            val_shift_limit=10,
                            p=1,
                        ),
                        A.CLAHE(clip_limit=2, p=1),
                    ],
                    p=0.5,
                ),
                A.GaussNoise(
                    std_range=(0.01, 0.05),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    noise_scale_factor=1.0,
                    p=0.2,
                ),
            ]

        # --- Normalization and tensor conversion ---
        base += [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        custom_transform = A.Compose(
            base,
            bbox_params=A.BboxParams(
                format="coco",  # we now correctly pass COCO-format boxes in/out
                label_fields=["labels"],
                min_visibility=0.3,
            ),
        )

        return custom_transform


class COCODetectionWrapper(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.dataset = CocoDetection(img_folder, ann_file)
        self.transforms = transforms

        # Map COCO category IDs (non-sequential) -> continuous label IDs
        label_ids = [cat["id"] for cat in self.dataset.coco.cats.values()]
        label_increment = 0
        if 0 in label_ids:
            logger.warning(
                "COCO category IDs contain 0, which is reserved for background."
            )
            logger.warning("Make sure that it is properly handled in your dataset.")
            logger.warning("Incrementing all category IDs by 1.")
            label_increment = 1  # +1 because 0 = background

        self.cat_id_to_label = {
            cat["id"]: idx + label_increment  # adding increment if needed
            for idx, cat in enumerate(self.dataset.coco.cats.values())
        }

    def __getitem__(self, idx):
        img, anns = self.dataset[idx]
        img = np.array(img, dtype=np.uint8)  # needed for Albumentations

        if len(anns) == 0:
            boxes_coco = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes_coco = np.array([ann["bbox"] for ann in anns], dtype=np.float32)
            boxes_coco = np.clip(boxes_coco, a_min=0, a_max=None)  # ensure non-negative
            labels = np.array(
                [self.cat_id_to_label[ann["category_id"]] for ann in anns],
                dtype=np.int64,
            )

        if self.transforms:
            sample = self.transforms(
                image=img, bboxes=boxes_coco.tolist(), labels=labels.tolist()
            )
            img = sample["image"]
            boxes_coco = np.array(sample["bboxes"], dtype=np.float32)
            labels = np.array(sample["labels"], dtype=np.int64)

        # Convert COCO to VOC format
        if boxes_coco.shape[0] > 0:
            boxes_voc = boxes_coco.copy()
            boxes_voc[:, 2] += boxes_voc[:, 0]  # x + w → x2
            boxes_voc[:, 3] += boxes_voc[:, 1]  # y + h → y2
        else:
            boxes_voc = np.zeros((0, 4), dtype=np.float32)

        target = {
            "boxes": torch.from_numpy(boxes_voc),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor(idx),
            "area": torch.from_numpy(
                (boxes_voc[:, 2] - boxes_voc[:, 0])
                * (boxes_voc[:, 3] - boxes_voc[:, 1])
            ),
            "iscrowd": torch.zeros((boxes_voc.shape[0],), dtype=torch.int64),
        }

        return img, target

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    return tuple(zip(*batch))
