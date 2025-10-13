# CORRECTED ORIGINAL CODE: Fixed both image size and labeler issues
model_name = "tf_efficientdet_d4"

from pathlib import Path
from typing import Union
from effdet.config import get_efficientdet_config
from effdet.bench import DetBenchTrain, DetBenchPredict
from ml_carbucks import DATA_DIR

config = get_efficientdet_config(model_name)

config.num_classes = 3

BATCH_SIZE = 1
# 🔧 FIX 1: Use model's expected image size instead of hardcoded 320
IMG_SIZE = config.image_size[0]  # This will be 640 for tf_efficientdet_d1

from effdet import EfficientDet

model = EfficientDet(config, pretrained_backbone=True)
model.reset_head(num_classes=config.num_classes)

# 🔧 FIX 2: Set create_labeler=True to auto-create the anchor labeler
bench = DetBenchTrain(model, create_labeler=True).cuda()
bench_config = bench.config

# Note: We don't need manual AnchorLabeler anymore since create_labeler=True
# from effdet.anchors import Anchors, AnchorLabeler
# labeler = AnchorLabeler(...)

from effdet import create_loader, create_dataset, create_evaluator
from effdet.anchors import Anchors, AnchorLabeler
from effdet.data.dataset_config import Coco2017Cfg
from effdet.data.parsers import CocoParserCfg, create_parser
from effdet.data.dataset import DetectionDatset
from collections import OrderedDict

# Allow limiting the number of images returned by the dataset
# `limit` takes an int: -1 means no limit, otherwise return up to `limit` examples
# `limit_mode`: 'first' -> take first N images, 'random' -> sample N images randomly
# `seed`: optional seed for reproducible random sampling


# Instead of torch.utils.data.Subset, create a dataset-like wrapper that
# preserves the original dataset type and API while only exposing the
# selected indices. This avoids type differences for downstream code.
class FilteredDataset:
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]

    # delegate attribute access for attributes not found on this wrapper
    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    # Explicitly expose parser and transform properties with setter delegation
    @property
    def parser(self):
        return self.base_dataset.parser

    @property
    def transform(self):
        return self.base_dataset.transform

    @transform.setter
    def transform(self, t):
        # When create_loader sets dataset.transform = ..., delegate to base dataset
        self.base_dataset.transform = t


def create_dataset_custom(
    name: str,
    img_dir: Union[str, Path],
    ann_file: Union[str, Path],
    limit: int = -1,
    limit_mode: str = "first",
    seed: int | None = None,
):

    datasets = OrderedDict()
    dataset_cfg = Coco2017Cfg()
    parser = CocoParserCfg(ann_filename=str(ann_file))
    dataset_cls = DetectionDatset
    dataset = dataset_cls(
        data_dir=img_dir,
        parser=create_parser(dataset_cfg.parser, cfg=parser),
    )

    # If limit is set and positive, create a FilteredDataset so the loader
    # only iterates over up to `limit` items. This limits images inside the dataset.
    if limit is not None and int(limit) > 0:
        n = min(int(limit), len(dataset))
        if limit_mode == "random":
            import random

            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), n)
        else:
            # default 'first' behaviour
            indices = list(range(n))
        dataset = FilteredDataset(dataset, indices)

    datasets[name] = dataset
    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]


train_dataset = create_dataset_custom(
    name="train",
    ann_file=DATA_DIR / "car_dd" / "instances_train.json",
    img_dir=DATA_DIR / "car_dd" / "images" / "train",
    limit=-1,
)

train_loader = create_loader(
    dataset=train_dataset,
    input_size=IMG_SIZE,  # Now uses correct size (640)
    batch_size=BATCH_SIZE,
    is_training=False,
)
it = iter(train_loader)
batch = next(it)

import torch

optimizer = torch.optim.AdamW(bench.parameters(), lr=3e-3)
EPOCHS = 500
bench.train()
for epoch in range(EPOCHS):
    ll = 0.0
    cl = 0.0
    bl = 0.0
    for batch in train_loader:
        inputs, targets = batch
        output = bench(inputs, targets)  # ✅ This will now work!
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ll += loss.item()
        cl += output["class_loss"].item()
        bl += output["box_loss"].item()

    print(
        f"Epoch {epoch+1}/{EPOCHS}, Loss: {ll:.4f}, cls_loss: {cl:.4f}, box_loss: {bl:.4f}"
    )


# save bench.model
torch.save(bench.model.state_dict(), f"effdet_{model_name}.pth")
