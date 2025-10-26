"""
This is not working out well.
Hours spent trying to get CenterNet to work properly have not yielded good results.
What is happening is that it is not

Take this as a warning sign. If you consider trying to fix this then increment the hour counter.
Hours spent: 37

Damian
"""

import time
from collections import defaultdict
import datetime as dt
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch import optim  # noqa: F401
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, resnet101
from effdet.data import create_loader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import FeaturePyramidNetwork

from ml_carbucks.utils.coco import (  # noqa: F401
    CocoStatsEvaluator,
    create_dataset_custom,
)


IMG_SIZE = 320
BATCH_SIZE = 8
NUM_CLASSES = 3
EPOCHS = 2000
DATASET_LIMIT = None
RUNTIME = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = "resnet50"  # resnet50 or resnet101

# DEPRECATED simpler head version


class SimpleCenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Shared conv
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Separate heads
        self.heatmap = nn.Conv2d(256, num_classes, 1)
        self.wh = nn.Conv2d(256, 2, 1)
        self.offset = nn.Conv2d(256, 2, 1)
        self._init_weights()

    def _init_weights(self):
        # Heatmap bias init -> lower confidence initially
        self.heatmap.bias.data.fill_(-2.19)  # type: ignore

    def forward(self, x):
        feat = self.shared(x)
        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "wh": self.wh(feat),
            "offset": self.offset(feat),
        }


class SimpleCenterNet(nn.Module):
    def __init__(self, num_classes=3, backbone_name="resnet50", pretrained=True):
        super().__init__()

        assert (
            backbone_name == "resnet50"
        ), "Only resnet50 backbone is supported in this implementation."

        backbone = resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # C5 feature map
        self.head = SimpleCenterNetHead(2048, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


# --- Head ---
class AdvancedCenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(256, num_classes, 1)
        self.wh = nn.Conv2d(256, 2, 1)
        self.offset = nn.Conv2d(256, 2, 1)
        self._init_weights()

    def _init_weights(self):
        # Lower initial confidence for heatmap
        self.heatmap.bias.data.fill_(-2.19)  # type: ignore

    def forward(self, x):
        feat = self.shared(x)
        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "wh": self.wh(feat),
            "offset": self.offset(feat),
        }


# --- Backbone + FPN ---
class AdvancedCenterNet(nn.Module):
    def __init__(self, backbone_name: str, num_classes=3, pretrained=True):
        super().__init__()
        if backbone_name == "resnet50":
            backbone = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        elif backbone_name == "resnet101":
            backbone = resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError("Unsupported backbone")
        # Extract intermediate feature maps (C3, C4, C5)
        # These correspond to strides 8, 16, 32
        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )  # -> C2 (1/4)
        self.layer2 = backbone.layer2  # -> C3 (1/8)
        self.layer3 = backbone.layer3  # -> C4 (1/16)
        self.layer4 = backbone.layer4  # -> C5 (1/32)

        # FPN: combines C3, C4, C5 into rich multi-scale features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],  # true channel sizes for resnet50
            out_channels=256,
        )

        # Final CenterNet head
        self.head = AdvancedCenterNetHead(256, num_classes)

    def forward(self, x):
        # Extract feature maps
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Build FPN features
        features = self.fpn({"c3": c3, "c4": c4, "c5": c5})

        # Use highest resolution FPN output for CenterNet head
        fpn_out = features["c3"]  # usually the 1/8 scale feature

        return self.head(fpn_out)


class MediumCenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # One light shared conv block instead of two
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(128, num_classes, 1)
        self.wh = nn.Conv2d(128, 2, 1)
        self.offset = nn.Conv2d(128, 2, 1)
        self._init_weights()

    def _init_weights(self):
        # lower initial confidence for heatmap
        nn.init.constant_(self.heatmap.bias, -2.19)  # type: ignore

    def forward(self, x):
        feat = self.shared(x)
        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "wh": self.wh(feat),
            "offset": self.offset(feat),
        }


# --- Backbone (ResNet, no FPN) ---
class MediumCenterNet(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=3, pretrained=True):
        super().__init__()

        if backbone_name == "resnet50":
            backbone = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            in_channels = 1024  # we’ll use C4
        elif backbone_name == "resnet101":
            backbone = resnet101(weights="IMAGENET1K_V2" if pretrained else None)
            in_channels = 1024
        else:
            raise ValueError("Unsupported backbone")

        # Keep up to layer3 (stride 16) — balance detail vs semantic info

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        # Reduce channels before head
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            # nn.BatchNorm2d(256),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        self.head = MediumCenterNetHead(256, num_classes)

    def forward(self, x):
        feat = self.backbone(x)  # stride 16 features
        feat = self.reduce(feat)  # channel compression
        out = self.head(feat)
        return out


def focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    return -(pos_loss + neg_loss) / (num_pos + 1e-4)


def compute_loss(preds, targets):
    hm_loss = focal_loss(preds["heatmap"], targets["heatmap"])
    wh_loss = F.l1_loss(preds["wh"], targets["wh"], reduction="mean")
    off_loss = F.l1_loss(preds["offset"], targets["offset"], reduction="mean")
    return hm_loss + wh_loss + off_loss, (hm_loss, wh_loss, off_loss)


def decode_predictions(preds, conf_thresh=0.5, stride=32, K=100, nms_kernel=3):
    """
    preds: dict with 'heatmap', 'wh', 'offset'
    conf_thresh: minimum confidence to keep a detection
    stride: scaling factor from heatmap to original image
    K: max number of detections per class
    nms_kernel: size of max pooling for NMS
    """
    heatmap, wh, offset = preds["heatmap"], preds["wh"], preds["offset"]
    batch, cat, height, width = heatmap.size()

    preds_device = heatmap.device
    # --- NMS on heatmap ---
    pooled = F.max_pool2d(
        heatmap, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2
    )
    keep = (pooled == heatmap).float()
    heatmap = heatmap * keep

    boxes_list = []
    scores_list = []
    labels_list = []

    for b in range(batch):
        for c in range(cat):
            hm_flat = heatmap[b, c].view(-1)
            topk_scores, topk_inds = torch.topk(hm_flat, K)
            topk_mask = topk_scores > conf_thresh
            if topk_mask.sum() == 0:
                continue

            topk_scores = topk_scores[topk_mask]
            topk_inds = topk_inds[topk_mask]

            ys = (topk_inds // width).float()
            xs = (topk_inds % width).float()

            w = wh[b, 0].view(-1)[topk_inds]
            h = wh[b, 1].view(-1)[topk_inds]

            off_x = offset[b, 0].view(-1)[topk_inds]
            off_y = offset[b, 1].view(-1)[topk_inds]

            xs = xs + off_x
            ys = ys + off_y

            x1 = (xs - w / 2) * stride
            y1 = (ys - h / 2) * stride
            x2 = (xs + w / 2) * stride
            y2 = (ys + h / 2) * stride

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            boxes_list.append(boxes)
            scores_list.append(topk_scores)
            labels_list.append(torch.full_like(topk_scores, c, dtype=torch.int))

    if len(boxes_list) == 0:
        return (
            torch.empty((0, 4)).to(preds_device),
            torch.empty((0,)).to(preds_device),
            torch.empty((0,), dtype=torch.int).to(preds_device),
        )

    boxes = torch.cat(boxes_list, dim=0)
    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # --- apply standard NMS ---
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return boxes, scores, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = SimpleCenterNet(num_classes=3, backbone_name=MODEL_NAME, pretrained=True).to(
#     device
# )

FREEZE_BACKBONE = False


model = MediumCenterNet(
    backbone_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    pretrained=True,
).to(device)


# model = AdvancedCenterNet(
#     backbone_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True
# ).to(device)

backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if "head" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.Adam(
    [
        {
            "params": backbone_params,
            # "lr": 5e-4,
            "weight_decay": 1e-5,
        },  # lower LR for pretrained backbone
        {
            "params": head_params,
            # "lr": 9e-3,
            "weight_decay": 5e-4,
        },  # higher LR for new head
    ]
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# scheduler = None


def draw_gaussian(heatmap, cx, cy, sigma=1.0):
    """Draw a 2D Gaussian on the heatmap at subpixel location cx, cy"""
    w, h = heatmap.shape[1], heatmap.shape[0]

    x = torch.arange(0, w, device=heatmap.device).float()
    y = torch.arange(0, h, device=heatmap.device).float().view(-1, 1)

    g = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    heatmap = torch.max(heatmap, g)
    return heatmap


def encode_targets(boxes, labels, output_size, num_classes, stride):
    heatmap = torch.zeros((num_classes, *output_size), device=device)
    wh = torch.zeros((2, *output_size), device=device)
    offset = torch.zeros((2, *output_size), device=device)

    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / stride
        cy = (y1 + y2) / 2 / stride
        w = (x2 - x1) / stride
        h = (y2 - y1) / stride
        cx_int, cy_int = int(cx), int(cy)

        # mark the heatmap (you can add a small gaussian, here just set 1)
        heatmap[cls, cy_int, cx_int] = 1
        wh[:, cy_int, cx_int] = torch.tensor([w, h], device=device)
        offset[:, cy_int, cx_int] = torch.tensor(
            [cx - cx_int, cy - cy_int], device=device
        )

    return {"heatmap": heatmap, "wh": wh, "offset": offset}


def encode_targets_gaussian(boxes, labels, output_size, num_classes, stride, sigma=1.2):
    heatmap = torch.zeros((num_classes, *output_size), device=device)
    wh = torch.zeros((2, *output_size), device=device)
    offset = torch.zeros((2, *output_size), device=device)

    H, W = output_size
    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / stride
        cy = (y1 + y2) / 2 / stride
        w = (x2 - x1) / stride
        h = (y2 - y1) / stride
        cx_int, cy_int = int(cx), int(cy)

        # draw gaussian on heatmap
        heatmap[cls] = draw_gaussian(heatmap[cls], cx, cy, sigma=sigma)

        wh[:, cy_int, cx_int] = torch.tensor([w, h], device=device)
        offset[:, cy_int, cx_int] = torch.tensor(
            [cx - cx_int, cy - cy_int], device=device
        )

    return {"heatmap": heatmap, "wh": wh, "offset": offset}


train_dataset = create_dataset_custom(
    name="train",
    img_dir=Path("/home/bachelor/ml-carbucks/data/car_dd/images/train"),
    ann_file=Path("/home/bachelor/ml-carbucks/data/car_dd/instances_train.json"),
    limit=DATASET_LIMIT,
)

train_loader = create_loader(
    train_dataset,
    input_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    use_prefetcher=True,
    num_workers=4,
    pin_mem=False,
)

val_dataset = create_dataset_custom(
    name="val",
    img_dir=Path("/home/bachelor/ml-carbucks/data/car_dd/images/val"),
    ann_file=Path("/home/bachelor/ml-carbucks/data/car_dd/instances_val.json"),
    limit=DATASET_LIMIT,
)
val_loader = create_loader(
    val_dataset,
    input_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    use_prefetcher=True,
    num_workers=4,
    pin_mem=False,
)
# val_evaluator = CocoStatsEvaluator(dataset=val_dataset, pred_yxyx=False)

steps_per_epoch = len(train_loader)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[5e-4, 1e-3],  # peak LR for each param group
    total_steps=EPOCHS * steps_per_epoch,
    pct_start=0.2,  # fraction of cycle to increase LR
    anneal_strategy="cos",  # cosine decay
    div_factor=10,  # initial LR = max_lr/div_factor
    final_div_factor=100,  # final LR = max_lr/final_div_factor
)


# Determine output size / stride once
dummy_input = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
with torch.no_grad():
    pred = model(dummy_input)
target_h, target_w = pred["heatmap"].shape[2], pred["heatmap"].shape[3]
stride = IMG_SIZE // target_h

# Now in your training loop, use this stride / target_h / target_w
training_progress = defaultdict(list)
COMPUTE_CYCLE = 1
CONFIDENCE_THRESHOLD = 0.15
VERBOSE = False

for epoch in range(EPOCHS):
    DO_ADDITIONAL_COMPUTE = (epoch + 1) % COMPUTE_CYCLE == 0 or epoch == 0
    val_stats = [-1] * 12
    model.train()
    start_time = time.time()
    for imgs, targets in tqdm(
        train_loader, desc=f"E {epoch + 1}/{EPOCHS} | Tb", disable=not VERBOSE
    ):
        imgs = imgs.to(device)

        # Split targets and encode using precomputed stride/size
        split_targets = [
            {"boxes": targets["bbox"][i], "labels": targets["cls"][i].long()}
            for i in range(imgs.shape[0])
        ]
        batch_targets = [
            encode_targets(
                t["boxes"], t["labels"], (target_h, target_w), NUM_CLASSES, stride
            )
            for t in split_targets
        ]
        batch_hm = torch.stack([bt["heatmap"] for bt in batch_targets])
        batch_wh = torch.stack([bt["wh"] for bt in batch_targets])
        batch_off = torch.stack([bt["offset"] for bt in batch_targets])

        # Forward + backward
        pred = model(imgs)
        loss, (hm_loss, wh_loss, off_loss) = compute_loss(
            pred, {"heatmap": batch_hm, "wh": batch_wh, "offset": batch_off}
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    if DO_ADDITIONAL_COMPUTE:
        model.eval()
        all_gts = []
        all_preds = []
        metric = MeanAveragePrecision()
        with torch.no_grad():
            for i, (val_imgs, val_targets) in enumerate(val_loader):
                val_imgs = val_imgs.to(device)

                val_preds = model(val_imgs)

                for i in range(val_imgs.shape[0]):
                    val_boxes_i, val_scores_i, val_labels_i = decode_predictions(
                        {k: v[i : i + 1] for k, v in val_preds.items()},
                        conf_thresh=CONFIDENCE_THRESHOLD,
                        stride=stride,
                        K=100,
                    )

                    gt_boxes = val_targets["bbox"][i].cpu()
                    gt_labels = val_targets["cls"][i].cpu()
                    # filter out -1 labels (no object)
                    mask = gt_labels != -1
                    gt_boxes = gt_boxes[mask]
                    gt_labels = gt_labels[mask]
                    all_gts.append({"boxes": gt_boxes, "labels": gt_labels.long()})
                    if val_boxes_i.shape[0] == 0:
                        all_preds.append(
                            {
                                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                                "scores": torch.zeros((0,), dtype=torch.float32),
                                "labels": torch.zeros((0,), dtype=torch.int64),
                            }
                        )
                    else:
                        all_preds.append(
                            {
                                "boxes": val_boxes_i.cpu(),
                                "scores": val_scores_i.cpu(),
                                "labels": val_labels_i.cpu().long(),
                            }
                        )
        metric.update(all_preds, all_gts)
        val_res = metric.compute()

    end_time = time.time()
    training_progress["epoch"].append(epoch + 1)
    training_progress["time"].append(round(end_time - start_time))
    training_progress["loss"].append(round(loss.item(), 2))  # type: ignore
    training_progress["hm_loss"].append(round(hm_loss.item(), 2))  # type: ignore
    training_progress["wh_loss"].append(round(wh_loss.item(), 2))  # type: ignore
    training_progress["off_loss"].append(round(off_loss.item(), 2))  # type: ignore

    if DO_ADDITIONAL_COMPUTE:
        training_progress["val_map"].append(round(val_res["map"].item(), 2))  # type: ignore
        training_progress["val_map_50"].append(round(val_res["map_50"].item(), 2))  # type: ignore
        training_progress["val_map_75"].append(round(val_res["map_75"].item(), 2))  # type: ignore
    else:
        training_progress["val_map"].append(-1)
        training_progress["val_map_50"].append(-1)
        training_progress["val_map_75"].append(-1)

    pd.DataFrame(training_progress).to_csv(
        f"results/training/training_{MODEL_NAME}_{RUNTIME}.csv", index=False
    )
    if DO_ADDITIONAL_COMPUTE:
        print(
            f"Epoch {epoch + 1}"
            f" | Loss: {training_progress['loss'][-1]}"
            f" | hm: {training_progress['hm_loss'][-1]}"
            f" | wh: {training_progress['wh_loss'][-1]}"
            f" | off: {training_progress['off_loss'][-1]}"
            f" | val_map: {training_progress['val_map'][-1]}"
            f" | val_map_50: {training_progress['val_map_50'][-1]}"
            f" | val_map_75: {training_progress['val_map_75'][-1]}"
        )
