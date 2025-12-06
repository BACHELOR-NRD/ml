import json
from pathlib import Path
from typing import List, Tuple
from typing_extensions import Literal
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import random
from torchvision.datasets import CocoDetection
from PIL import ImageOps

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


def preprocess_images(
    images: List[np.ndarray], img_size: int
) -> Tuple[List[torch.Tensor], List[float], List[Tuple[int, int]]]:
    transform = create_transforms(is_training=False, img_size=img_size)

    preprocessed_samples = [
        transform(image=img, bboxes=[], labels=[]) for img in images
    ]
    preprocessed_images = [sample["image"] for sample in preprocessed_samples]

    scales = [img_size / max(img.shape[0], img.shape[1]) for img in images]
    original_sizes = [(img.shape[1], img.shape[0]) for img in images]  # (width, height)
    return preprocessed_images, scales, original_sizes


def create_transforms(
    is_training: bool,
    img_size: int,
    affine: bool = True,
    flip: bool = True,
    crop: bool = True,
    noise: bool = True,
    color_jitter: bool = True,
) -> A.Compose:

    base = [
        # Always resize and pad to square input
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,  # constant padding
            fill=(0, 0, 0),
            position="top_left",
        ),
    ]

    if is_training:
        # --- Spatial augmentations (geometry) ---
        if affine:
            base += [
                A.Affine(
                    translate_percent={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                    },  # ±5% shift
                    scale=(0.9, 1.1),  # ±10% zoom
                    rotate=(-10, 10),  # ±10 degrees rotation
                    fit_output=False,
                    fill=(0, 0, 0),  # fill color for borders
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    p=0.7,
                )
            ]
        if flip:
            base += [A.HorizontalFlip(p=0.5)]
        if crop:
            base += [A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, p=0.3)]
        # --- Photometric augmentations (color) ---
        if color_jitter:
            base += [
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
                )
            ]
        if noise:
            base += [
                A.GaussNoise(
                    std_range=(0.01, 0.05),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    noise_scale_factor=1.0,
                    p=0.2,
                )
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


def simple_transform() -> A.Compose:
    return A.Compose([ToTensorV2()])


class COCODetectionWrapper(Dataset):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        exif_aware: bool = False,
        debugging: bool = False,
        format: Literal["xyxy", "yxyx", "xywh"] = "xyxy",
        placeholders: int = 0,
    ):
        self.dataset = CocoDetection(img_folder, ann_file)
        self.transforms = transforms
        self.exif_aware = exif_aware
        self.debugging = debugging
        self.format = format
        self.placeholders = placeholders

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
            cat["id"]: cat["id"] + label_increment
            for _, cat in enumerate(self.dataset.coco.cats.values())
        }

        anns = json.load(open(ann_file, "r"))
        self.img_id_to_path = {}
        for img_info in anns["images"]:
            img_id = img_info["id"]
            img_filename = img_info["file_name"]
            self.img_id_to_path[img_id] = str(Path(img_folder) / img_filename)

    def __getitem__(self, idx):
        img, anns = self.dataset[idx]

        # NOTE: Modern images may have EXIF orientation data that needs to be handled
        # without this, some images may be loaded in wrong orientation
        if self.exif_aware:
            img = ImageOps.exif_transpose(img)

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

        org_h, org_w = img.shape[0], img.shape[1]
        if self.transforms:
            sample = self.transforms(
                image=img, bboxes=boxes_coco.tolist(), labels=labels.tolist()
            )
            img = sample["image"]
            boxes_coco = np.array(sample["bboxes"], dtype=np.float32)
            labels = np.array(sample["labels"], dtype=np.int64)
        else:
            # If no transforms are applied, make sure that img is a np array with proper permutations
            # img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
            pass

        # Convert COCO to VOC format
        if self.format == "xyxy":
            if boxes_coco.shape[0] > 0:
                boxes_voc = boxes_coco.copy()
                boxes_voc[:, 2] += boxes_voc[:, 0]  # x + w → x2
                boxes_voc[:, 3] += boxes_voc[:, 1]  # y + h → y2
            else:
                boxes_voc = np.zeros((0, 4), dtype=np.float32)
        elif self.format == "yxyx":
            if boxes_coco.shape[0] > 0:
                boxes_voc = boxes_coco.copy()
                boxes_voc[:, [0, 1, 2, 3]] = boxes_voc[:, [1, 0, 3, 2]]  # swap x<->y
                boxes_voc[:, 2] += boxes_voc[:, 0]  # x + w → x2
                boxes_voc[:, 3] += boxes_voc[:, 1]  # y + h → y2
            else:
                boxes_voc = np.zeros((0, 4), dtype=np.float32)
        elif self.format == "xywh":
            boxes_voc = boxes_coco
        else:
            raise ValueError(f"Unsupported box format: {self.format}")

        boxes = torch.from_numpy(boxes_voc)
        labels = torch.from_numpy(labels)
        area = torch.from_numpy(
            (boxes_voc[:, 2] - boxes_voc[:, 0]) * (boxes_voc[:, 3] - boxes_voc[:, 1])
        )
        iscrowd = torch.zeros((boxes_voc.shape[0],), dtype=torch.int64)
        img_scale = torch.tensor(
            max(org_h, org_w) / max(img.shape[0], img.shape[1]), dtype=torch.float32
        )

        if self.placeholders > 0:
            # include placeholdes with -1.0 values everywhere, make sure that each tensor has size placeholders
            num_boxes = boxes.shape[0]
            if num_boxes < self.placeholders:
                pad_size = self.placeholders - num_boxes
                boxes = torch.cat([boxes, torch.full((pad_size, 4), -1.0)], dim=0)
                labels = torch.cat(
                    [labels, torch.full((pad_size,), -1, dtype=torch.int64)], dim=0
                )
                area = torch.cat([area, torch.full((pad_size,), -1.0)], dim=0)
                iscrowd = torch.cat(
                    [iscrowd, torch.full((pad_size,), -1, dtype=torch.int64)], dim=0
                )
                boxes = boxes[: self.placeholders]
                labels = labels[: self.placeholders]
                area = area[: self.placeholders]
                iscrowd = iscrowd[: self.placeholders]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx),
            "area": area,
            "iscrowd": iscrowd,
            "img_scale": img_scale,
            "img_size": torch.tensor([org_w, org_h], dtype=torch.int64),
        }
        if self.debugging:
            # NOTE: image path is useful for debugging purposes
            target["image_path"] = self.img_id_to_path[self.dataset.ids[idx]]

        return img, target

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    return tuple(zip(*batch))


def create_clean_loader(
    datasets: list[tuple[str | Path, str | Path]],
    shuffle: bool,
    batch_size: int,
    transforms: A.Compose | None,
    exif_aware: bool = False,
    debugging: bool = False,
    format: Literal["xyxy", "yxyx", "xywh"] = "xyxy",
    placeholders: int = 0,
) -> DataLoader:
    """A function that creates a neat loader for image datasets.
    If no transforms are provided, images will be loaded as np.arrays in format HWC with dtype uint8.
    """

    # NOTE: Something here is causing the Jupyter notebooks to hang in debugging. Something perhaps in next iter call?
    all_datasets = []
    for img_folder, ann_file in datasets:
        ds = COCODetectionWrapper(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            exif_aware=exif_aware,
            debugging=debugging,
            format=format,
            placeholders=placeholders,
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


# ---- EXPERIMENTAL FUNCTIONS ----


def create_train_val_loaders(
    datasets: list[tuple[str | Path, str | Path]],
    split: float = 0.8,
    seed: int = 42,
    batch_size: int = 8,
    transforms: A.Compose | None = None,
    exif_aware: bool = False,
    debugging: bool = False,
    format: Literal["xyxy", "yxyx", "xywh"] = "xyxy",
    placeholders: int = 0,
    num_workers: int | None = None,
    pin_mem: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from dataset paths.

    Args:
        datasets: list of (img_folder, ann_file) tuples. Multiple datasets are concatenated.
        split: fraction of data to use for training (0.0-1.0).
        seed: RNG seed for reproducible split and shuffling.
        batch_size: DataLoader batch size.
        transforms: Albumentations transforms applied to each dataset sample.
        exif_aware: whether to honor EXIF orientation when loading images.
        debugging: enable debugging outputs in the wrapper.
        format: box format returned by wrapper.
        placeholders: pad target tensors to this many boxes (per-sample).
        num_workers: number of DataLoader workers (default: max(4, batch_size//2)).
        pin_mem: pass pin_memory to DataLoader.

    Returns:
        (train_loader, val_loader)
    """

    all_datasets = []
    for img_folder, ann_file in datasets:
        ds = COCODetectionWrapper(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            exif_aware=exif_aware,
            debugging=debugging,
            format=format,
            placeholders=placeholders,
        )
        all_datasets.append(ds)

    combined = ConcatDataset(all_datasets)
    total = len(combined)
    if total == 0:
        raise ValueError("No data found in provided datasets")

    indices = list(range(total))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    split_point = int(total * float(split))
    train_idx = indices[:split_point]
    val_idx = indices[split_point:]

    train_subset = Subset(combined, train_idx)
    val_subset = Subset(combined, val_idx)

    if num_workers is None:
        num_workers = max(4, batch_size // 2)

    # deterministic shuffling using a Generator seeded with `seed`
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        generator=g,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader


def split_and_write_coco_annotations(
    datasets: list[tuple[str | Path, str | Path]],
    split: float = 0.8,
    seed: int = 42,
    output_dir: Path | str | None = None,
    prefix: str = "auto",
) -> tuple[Path, Path]:
    """Combine COCO annotation files from `datasets`, split images into train/val and
    write two COCO-format JSON files with the given `prefix`.

    The function will:
      - read all annotation files listed in `datasets`
      - merge categories (assumes consistent category schema across files)
      - renumber image and annotation ids to produce a clean combined dataset
      - split images deterministically using `seed` and `split` fraction
      - write two files: {prefix}_train.json and {prefix}_val.json into `output_dir`

    Returns: (train_json_path, val_json_path)
    """
    if len(datasets) == 0:
        raise ValueError("No datasets provided")

    # Read and combine
    combined_images = []
    combined_annotations = []
    combined_categories: dict[int, dict] = {}
    img_global_id = 1
    ann_global_id = 1
    image_id_map = {}  # (ann_file, original_image_id) -> new_image_id

    for img_folder, ann_file in datasets:
        ann_file = str(ann_file)
        with open(ann_file, "r") as f:
            data = json.load(f)

        # merge categories (keep dict by id -> category entry)
        for cat in data.get("categories", []):
            cid = int(cat["id"])
            if cid not in combined_categories:
                combined_categories[cid] = cat

        # map images
        for img in data.get("images", []):
            orig_id = int(img.get("id"))
            new_id = img_global_id
            image_id_map[(ann_file, orig_id)] = new_id

            # copy image entry but replace id and keep file_name
            new_img = dict(img)
            new_img["id"] = new_id
            combined_images.append(new_img)
            img_global_id += 1

        # map annotations
        for ann in data.get("annotations", []):
            orig_img_id = int(ann.get("image_id"))
            new_img_id = image_id_map.get((ann_file, orig_img_id))
            if new_img_id is None:
                # annotation references missing image; skip
                continue
            new_ann = dict(ann)
            new_ann["id"] = ann_global_id
            new_ann["image_id"] = new_img_id
            combined_annotations.append(new_ann)
            ann_global_id += 1

    total_images = len(combined_images)
    if total_images == 0:
        raise ValueError("Combined annotation contains 0 images")

    # Shuffle and split images deterministically
    indices = list(range(total_images))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_point = int(total_images * float(split))
    train_idx = set(indices[:split_point])
    val_idx = set(indices[split_point:])

    # partition images
    train_images = [img for i, img in enumerate(combined_images) if i in train_idx]
    val_images = [img for i, img in enumerate(combined_images) if i in val_idx]

    # partition annotations by new image ids
    train_image_ids = {img["id"] for img in train_images}
    val_image_ids = {img["id"] for img in val_images}

    train_annotations = [
        ann for ann in combined_annotations if ann["image_id"] in train_image_ids
    ]
    val_annotations = [
        ann for ann in combined_annotations if ann["image_id"] in val_image_ids
    ]

    # assemble output dicts
    categories_list = list(combined_categories.values())

    train_dict = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories_list,
    }

    val_dict = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories_list,
    }

    # pick output dir
    if output_dir is None:
        # put into directory of first annotation file
        first_ann_dir = Path(datasets[0][1]).resolve().parent
        output_dir = first_ann_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"{prefix}_train.json"
    val_path = output_dir / f"{prefix}_val.json"

    with open(train_path, "w") as f:
        json.dump(train_dict, f)
    with open(val_path, "w") as f:
        json.dump(val_dict, f)

    return train_path, val_path
