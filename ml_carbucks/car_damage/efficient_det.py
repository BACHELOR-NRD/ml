# from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
# from effdet.efficientdet import HeadNet
# from torch.utils.data import DataLoader
# from effdet.data import CocoDetection
# from torchvision import transforms
# import torch
# model_name = "tf_efficientdet_d4"  # or d0–d7
# config = get_efficientdet_config(model_name)
# config.num_classes = 3
# config.image_size = (640, 640)
# model = EfficientDet(config, pretrained_backbone=True)
# model.class_net = HeadNet(config, num_outputs=config.num_classes)
# bench = DetBenchTrain(model, config)
# bench = bench.cuda()
# train_dataset = CocoDetection(
#     "dataset/images/train", "dataset/train.json", transform=transforms.ToTensor()
# )
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
# optimizer = torch.optim.AdamW(bench.parameters(), lr=1e-4)
# for epoch in range(10):
#     bench.train()
#     for imgs, targets in train_loader:
#         imgs = imgs.cuda()
#         targets = [
#             {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()}
#             for t in targets
#         ]
#         loss = bench(imgs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch}: loss={loss.item():.4f}")
# torch.save(model.state_dict(), "efficientdet_d1_damage.pth")


from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection  # use torchvision, not effdet.data
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
from ml_carbucks import DATA_CAR_DD_DIR

# 1. Create config & model
model_name = "tf_efficientdet_d0"  # start with small model
config = get_efficientdet_config(model_name)
config.num_classes = 3  # your classes
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
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ]
)
train_dataset = CocoDetection(
    root=DATA_CAR_DD_DIR / "images" / "train",
    annFile=str(DATA_CAR_DD_DIR / "instances_train.json"),
    transform=transform,
)


def coco_to_effdet_targets(coco_dataset):
    """
    Converts COCO annotations to effdet-compatible targets.
    Returns a list of dicts, one per image in coco_dataset.
    """
    # Group annotations by image_id
    targets = {
        "bbox": [],
        "cls": [],
    }

    for t in coco_dataset:
        bboxes = []
        labels = []
        for ann in t:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if len(bboxes) == 0:
            targets["bbox"].append(torch.zeros((0, 4), dtype=torch.float32))
            targets["cls"].append(torch.zeros((0,), dtype=torch.int64))
        else:
            targets["bbox"].append(torch.tensor(bboxes, dtype=torch.float32))
            targets["cls"].append(torch.tensor(labels, dtype=torch.int64))
    return targets


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = coco_to_effdet_targets([b[1] for b in batch])
    return imgs, targets


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)

# 5. Training loop
optimizer = torch.optim.AdamW(bench.parameters(), lr=1e-4)


for epoch in range(10):
    bench.train()
    for batch in train_loader:
        imgs, targets = batch

        imgs = imgs.cuda()

        # transform targets to move tensors to cuda
        targets = {
            "bbox": [t.cuda() for t in targets["bbox"]],
            "cls": [t.cuda() for t in targets["cls"]],
        }
        output = bench(imgs, targets)
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} loss: {loss.item()}")  # type: ignore

# 6. Save model weights (just the model)
torch.save(model.state_dict(), "efficientdet_d0_damage.pth")

# 7. Inference (wrap with DetBenchPredict)
predict_bench = DetBenchPredict(model).cuda()
predict_bench.eval()
# Example on one image:
# img_tensor = ToTensor()(Image.open(...)).unsqueeze(0).cuda()
# detections = predict_bench(img_tensor)
