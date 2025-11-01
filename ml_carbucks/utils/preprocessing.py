from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import CocoDetection

from ml_carbucks.utils.logger import setup_logger

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


def create_loader(
    datasets: list[tuple[str | Path, str | Path]],
    shuffle: bool,
    batch_size: int,
    transforms: A.Compose | None,
) -> DataLoader:

    all_datasets = []
    for img_folder, ann_file in datasets:
        ds = COCODetectionWrapper(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
        )
        all_datasets.append(ds)

    combined_dataset = ConcatDataset(all_datasets)

    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(4, batch_size // 2),
        collate_fn=collate_fn,
    )

    return loader
