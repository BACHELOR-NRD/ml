import datetime as dt

import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import (  # noqa: F401
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from ml_carbucks import DATA_CAR_DD_DIR, RESULTS_DIR
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.training import ResultSaver

IMG_SIZE = 512
BATCH_SIZE = 16
NUM_CLASSES = 4  # background + 3 object classes
RUNTIME = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class COCODetectionWrapper(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.dataset = CocoDetection(img_folder, ann_file)
        self.transforms = transforms

        # Map COCO category IDs (non-sequential) -> continuous label IDs
        self.cat_id_to_label = {
            cat["id"]: idx + 1  # +1 because 0 = background
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


def create_transforms(is_training: bool) -> A.Compose:

    base = [
        # Always resize and pad to square input
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
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
            A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, p=0.3),
            # --- Photometric augmentations (color) ---
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1
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


# --- Dataset ---
train_dataset = COCODetectionWrapper(
    img_folder=DATA_CAR_DD_DIR / "images" / "train",
    ann_file=DATA_CAR_DD_DIR / "instances_train.json",
    transforms=create_transforms(is_training=True),
)

val_dataset = COCODetectionWrapper(
    img_folder=DATA_CAR_DD_DIR / "images" / "val",
    ann_file=DATA_CAR_DD_DIR / "instances_val.json",
    transforms=create_transforms(is_training=False),
)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,  # 2–8 is typical, memory permitting
    shuffle=True,
    num_workers=BATCH_SIZE // 2,  # adjust based on your CPU
    pin_memory=True,
    collate_fn=collate_fn,  # crucial
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=BATCH_SIZE // 2,
    pin_memory=True,
    collate_fn=collate_fn,
)


logger = setup_logger("faster_rcnn")


logger.info("Running Faster R-CNN training demo")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model = model.to(device)

# --- Optimizer ---
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "backbone" in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": 5e-5, "weight_decay": 1e-5},
        {"params": head_params, "lr": 5e-4, "weight_decay": 1e-4},
    ]
)

EPOCHS = 50
# --- Scheduler (optional) ---
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


saver = ResultSaver(
    res_dir=RESULTS_DIR / "faster_rcnn",
    name=f"training_results_{RUNTIME}",
    metadata={
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__ if scheduler else "None",
        "backbone": model.backbone.__class__.__name__,
    },
)

# --- Training loop ---
num_epochs = EPOCHS
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for imgs, targets in tqdm(train_loader):

        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()

        total_loss += loss.item()  # type: ignore

    if scheduler:
        scheduler.step()

    # --- Validation on training data (resized) ---
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader):
            imgs = list(img.to(device) for img in imgs)
            outputs = model(imgs)

            # Prepare targets in expected dict format
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            metric.update(outputs_cpu, targets_cpu)

    val_res = metric.compute()
    metric.reset()
    logger.info(
        f"Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss:.4f} | val_map: {val_res['map'].item():.4f}"
    )
    saver.save(
        epoch=epoch + 1,
        loss=total_loss,
        val_map=val_res["map"].item(),
        val_map_50=val_res["map_50"].item(),
    ).plot(show=False)
