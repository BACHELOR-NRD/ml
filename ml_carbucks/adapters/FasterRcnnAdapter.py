from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.datasets import CocoDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_PREDICTION,
    BaseDetectionAdapter,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.result_saver import ResultSaver

logger = setup_logger(__name__)


def create_transforms(is_training: bool, img_size: int) -> A.Compose:

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


@dataclass
class FasterRcnnAdapter(BaseDetectionAdapter):

    weights: str | Path = "DEFAULT"

    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay_backbone: float = 1e-5
    weight_decay_head: float = 1e-4

    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model{suffix}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def predict(self, images: List[torch.Tensor]) -> List[ADAPTER_PREDICTION]:
        raise NotImplementedError("Predict method is not yet implemented.")

    def setup(self) -> "FasterRcnnAdapter":
        logger.debug("Creating Faster R-CNN model...")

        img_size = self.img_size

        weights = self.weights

        if weights == "DEFAULT":
            self.model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                min_size=img_size,
                max_size=img_size,
            )
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, len(self.classes) + 1  # +1 for background
            )
        elif weights is not None and Path(weights).is_file():
            self.model = fasterrcnn_resnet50_fpn(
                pretrained=False, num_classes=len(self.classes) + 1
            )
            checkpoint = torch.load(weights, map_location=self.device)  # type: ignore
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(
                "Weights must be 'DEFAULT' or a valid path to a checkpoint."
            )

        self.model.to(self.device)

        return self

    def _create_optimizer(self):
        lr1 = self.lr_backbone
        lr2 = self.lr_head
        weight_decay1 = self.weight_decay_backbone
        weight_decay2 = self.weight_decay_head

        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            elif "roi_heads" in name:
                head_params.append(param)

        params = [
            {"params": backbone_params, "lr": lr1, "weight_decay": weight_decay1},
            {"params": head_params, "lr": lr2, "weight_decay": weight_decay2},
        ]

        return torch.optim.AdamW(params)

    def _create_loader(
        self, datasets: List[Tuple[str | Path, str | Path]], is_training: bool
    ) -> DataLoader:
        batch_size = self.batch_size
        img_size = self.img_size

        all_datasets = []
        for img_dir, ann_file in datasets:
            ds = COCODetectionWrapper(
                img_folder=img_dir,
                ann_file=ann_file,
                transforms=create_transforms(
                    is_training=is_training, img_size=img_size
                ),
            )
            all_datasets.append(ds)

        combined_dataset = ConcatDataset(all_datasets)

        loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,  # 2–8 is typical, memory permitting
            shuffle=is_training,
            num_workers=max(4, batch_size // 2),  # adjust based on your CPU
            pin_memory=True,
            collate_fn=collate_fn,  # crucial
        )

        return loader

    def fit(self, datasets: List[Tuple[str | Path, str | Path]]) -> "FasterRcnnAdapter":
        logger.info("Starting training...")

        epochs = self.epochs

        loader = self._create_loader(datasets, is_training=True)

        optimizer = self._create_optimizer()

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")

            _ = self.train_epoch(optimizer, loader)

        return self

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
    ) -> float:
        self.model.train()

        total_loss = 0.0
        for imgs, targets in tqdm(loader, desc="Training", unit="batch"):
            imgs = list(img.to(self.device) for img in imgs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()

            total_loss += loss.item()  # type: ignore

        return total_loss

    def debug(
        self,
        train_datasets: List[Tuple[str | Path, str | Path]],
        val_datasets: List[Tuple[str | Path, str | Path]],
        results_path: str | Path,
        results_name: str,
    ) -> None:
        logger.info("Debugging training and evaluation loops...")
        epochs = self.epochs
        train_loader = self._create_loader(train_datasets, is_training=True)
        optimizer = self._create_optimizer()
        saver = ResultSaver(results_path, name=results_name)
        for epoch in range(1, epochs + 1):
            logger.info(f"Debug Epoch {epoch}/{epochs}")
            total_loss = self.train_epoch(optimizer, train_loader)
            val_metrics = self.evaluate(val_datasets)
            saver.save(
                epoch=epoch,
                loss=total_loss,
                val_map=val_metrics["map_50_95"],
                val_map_50=val_metrics["map_50"],
            )
            logger.info(
                f"Debug Epoch {epoch}/{epochs} - Loss: {total_loss}, Val MAP: {val_metrics['map_50_95']}"
            )

        saver.plot(show=False)

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> Dict[str, float]:
        logger.info("Starting evaluation...")
        self.model.eval()

        loader = self._create_loader(datasets, is_training=False)

        metric = MeanAveragePrecision()
        with torch.no_grad():
            for imgs, targets in loader:
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

    def clone(self) -> "FasterRcnnAdapter":
        return FasterRcnnAdapter(
            classes=deepcopy(self.classes),
            weights=self.weights,
            img_size=self.img_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr_backbone=self.lr_backbone,
            lr_head=self.lr_head,
            weight_decay_backbone=self.weight_decay_backbone,
            weight_decay_head=self.weight_decay_head,
        )
