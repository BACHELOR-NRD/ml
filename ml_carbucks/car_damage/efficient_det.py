from functools import partial
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection  # use torchvision, not effdet.data
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from ml_carbucks import DATA_CAR_DD_DIR
from copy import deepcopy
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 1. Create config & model
model_name = "tf_efficientdet_d0"  # start with small model
config = get_efficientdet_config(model_name)
config.num_classes = 3  # your classes
config.max_det_per_image = 20
config.image_size = (320, 320)  # or square int if config expects single int
# Optionally set other config fields:
# config.norm_kwargs = dict(eps=1e-3, momentum=0.01)

# 2. Build model
model = EfficientDet(config, pretrained_backbone=True)
model.class_net = HeadNet(config, num_outputs=config.num_classes)
# At this point, model and config should be consistent

# 3. Wrap for training
bench = DetBenchTrain(model).cuda()

# 4. Prepare dataset using torchvision
transform = transforms.Compose(
    [
        # transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ]
)
train_dataset = CocoDetection(
    root=DATA_CAR_DD_DIR / "images" / "train",
    annFile=str(DATA_CAR_DD_DIR / "instances_train.json"),
    transform=deepcopy(transform),
)
val_dataset = CocoDetection(
    root=DATA_CAR_DD_DIR / "images" / "val",
    annFile=str(DATA_CAR_DD_DIR / "instances_val.json"),
    transform=deepcopy(transform),
)


def coco_to_effdet_targets(coco_dataset, scales: list, pad_xs: list, pad_ys: list):
    """
    Converts COCO annotations to effdet-compatible targets.
    Returns a list of dicts, one per image in coco_dataset.
    """
    # Group annotations by image_id
    targets = {
        "bbox": [],
        "cls": [],
    }

    for i, t in enumerate(coco_dataset):
        bboxes = []
        labels = []
        for ann in t:
            x, y, w, h = ann["bbox"]

            bboxes.append(
                [
                    x * scales[i] + pad_xs[i],
                    y * scales[i] + pad_ys[i],
                    (x + w) * scales[i] + pad_xs[i],
                    (y + h) * scales[i] + pad_ys[i],
                ]
            )
            labels.append(ann["category_id"])

        if len(bboxes) == 0:
            targets["bbox"].append(torch.zeros((0, 4), dtype=torch.float32))
            targets["cls"].append(torch.zeros((0,), dtype=torch.int64))
        else:
            targets["bbox"].append(torch.tensor(bboxes, dtype=torch.float32))
            targets["cls"].append(torch.tensor(labels, dtype=torch.int64))
    return targets


def resize_with_padding_tensor(
    img_tensor: torch.Tensor, img_size: Optional[int] = None
) -> Tuple[torch.Tensor, float, int, int]:
    """
    Efficiently resize [C,H,W] tensor to img_size x img_size with aspect ratio preserved,
    adding padding. Returns new tensor, scale, pad_x, pad_y.
    """
    C, H, W = img_tensor.shape
    if img_size is None:
        return img_tensor, 1.0, 0, 0

    scale = img_size / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)

    # Resize in a single step
    img_tensor = F.interpolate(
        img_tensor[None], size=(new_H, new_W), mode="bilinear", align_corners=False
    )[0]

    pad_x = (img_size - new_W) // 2
    pad_y = (img_size - new_H) // 2

    # Pad: (left, right, top, bottom)
    new_img = F.pad(
        img_tensor, (pad_x, img_size - new_W - pad_x, pad_y, img_size - new_H - pad_y)
    )

    return new_img, scale, pad_x, pad_y


def collate_fn(batch, img_size):

    scales = []
    pad_xs = []
    pad_ys = []
    imgs = []
    for i, p in enumerate(batch):
        img, scale, pad_x, pad_y = resize_with_padding_tensor(p[0], img_size)
        imgs.append(img)
        scales.append(scale)
        pad_xs.append(pad_x)
        pad_ys.append(pad_y)

    imgs = torch.stack(imgs)
    targets = coco_to_effdet_targets([p[1] for p in batch], scales, pad_xs, pad_ys)

    return imgs, targets


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    collate_fn=partial(collate_fn, img_size=config.image_size[0]),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    collate_fn=partial(collate_fn, img_size=config.image_size[0]),
)

# 5. Training loop
optimizer = torch.optim.AdamW(bench.parameters(), lr=1e-4)


def move2cuda(imgs, targets, is_val: bool = False):
    new_imgs = imgs.cuda()
    new_targets = {
        "bbox": [t.cuda() for t in targets["bbox"]],
        "cls": [t.cuda() for t in targets["cls"]],
    }

    if is_val:
        new_targets["img_size"] = None  # type: ignore
        new_targets["img_scale"] = None  # type: ignore

    return new_imgs, new_targets


def ppp(tensor, bboxes_xyxy):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    for box in bboxes_xyxy:
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
        )
        plt.gca().add_patch(rect)
    plt.axis("off")
    plt.show()


map_metric = MeanAveragePrecision()
for epoch in range(50):
    bench.train()

    print(f"Epoch {epoch} starting...")
    for l_imgs, l_targets in train_loader:
        imgs, targets = move2cuda(l_imgs, l_targets, is_val=False)
        output = bench(imgs, targets)
        ppp(imgs[0], targets["bbox"][0])
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Evaluating...")
    bench.eval()
    map_metric.reset()
    val_loss = 0.0
    confidence_threshold = 0.5

    with torch.no_grad():
        for l_imgs, l_targets in val_loader:
            imgs, targets = move2cuda(l_imgs, l_targets, is_val=True)
            output = bench(imgs, targets)
            val_loss += output["loss"].item()

            detections = output["detections"]
            B = detections.shape[0]
            batch_preds = []
            batch_targets = []

            for i in range(B):
                det = detections[i]

                mask = det[:, 4] >= confidence_threshold
                det = det[mask]

                batch_preds.append(
                    {
                        "boxes": det[:, :4],
                        "scores": det[:, 4],
                        "labels": det[:, 5].long(),
                    }
                )
                batch_targets.append(
                    {
                        "boxes": targets["bbox"][i],
                        "labels": targets["cls"][i].long(),
                    }
                )

            map_metric.update(batch_preds, batch_targets)

    val_loss /= len(val_loader)
    map_res = map_metric.compute()

    print(f"Epoch {epoch} loss: {loss.item():<.4f} val_loss: {val_loss:<.4f} map50: {map_res['map_50']:<.4f}")  # type: ignore


# 6. Save model weights (just the model)
torch.save(model.state_dict(), "efficientdet_d0_damage.pth")
