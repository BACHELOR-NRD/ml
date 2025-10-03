import os
import json
import warnings

import torch
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    TrainingArguments,
    Trainer,
)
from PIL import Image
from datasets import load_dataset, DatasetDict, Dataset
from torchvision import transforms as T

from ml_carbucks import DATA_CAR_DD_DIR

# Path to your data
data_dir = str(DATA_CAR_DD_DIR)


# Load annotations
ds = load_dataset(
    "json",
    data_files={
        "train": f"{data_dir}/instances_train.json",
        "val": f"{data_dir}/instances_val.json",
        "test": f"{data_dir}/instances_test.json",
    },
    field="annotations",
)


# Map image_id -> file_name
def load_image_info(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)
    return {img["id"]: img["file_name"] for img in coco["images"]}


# Group annotations per image (store paths only)
def group_annotations_lazy(ds_split, images_dir, image_id_to_file):
    grouped = {}
    for ann in ds_split:
        img_id = ann["image_id"]
        if img_id not in grouped:
            grouped[img_id] = {
                "image_path": os.path.join(images_dir, image_id_to_file[img_id]),
                "bboxes": [],
                "labels": [],
            }
        grouped[img_id]["bboxes"].append(ann["bbox"])
        grouped[img_id]["labels"].append(ann["category_id"])

    examples = []
    for info in grouped.values():
        examples.append(
            {
                "image_path": info["image_path"],
                "bboxes": info["bboxes"],
                "labels": info["labels"],
            }
        )
    return Dataset.from_list(examples)


# Build DatasetDict
def get_grouped_dataset():
    split_dirs = {"train": "train", "val": "val", "test": "test"}
    grouped_datasets = {}
    for split, folder in split_dirs.items():
        images_dir = os.path.join(data_dir, "images", folder)
        image_file_map = load_image_info(f"{data_dir}/instances_{split}.json")
        grouped_ds = group_annotations_lazy(ds[split], images_dir, image_file_map)
        grouped_datasets[split] = grouped_ds

    return DatasetDict(grouped_datasets)


grouped_ds = get_grouped_dataset()
print("Dataset ready. Example:", grouped_ds["train"][0])


processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50", size={"shortest_edge": 320, "longest_edge": 320}
)
transform = T.Compose([T.ToTensor()])  # can add resizing if needed


def collate_fn(batch):
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    targets = []

    for idx, item in enumerate(batch):
        annots = []
        for bbox, label in zip(item["bboxes"], item["labels"]):
            x, y, w, h = bbox
            annots.append(
                {"bbox": bbox, "category_id": label, "iscrowd": 0, "area": w * h}
            )
        targets.append({"image_id": idx, "annotations": annots})

    encodings = processor(images=images, annotations=targets, return_tensors="pt")
    return encodings


# determine number of object classes
all_labels = [label for ex in grouped_ds["train"] for label in ex["labels"]]
num_classes = max(all_labels) + 1
print(f"Number of classes: {num_classes}")

# Suppress PyTorch meta tensor warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.nn.modules.module"
)

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
    torch_dtype=torch.float32,  # Explicitly specify dtype to avoid meta tensor issues
)

training_args = TrainingArguments(
    output_dir="./outputs/detr_car_dd",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=grouped_ds["train"],
    eval_dataset=grouped_ds["val"],
    data_collator=collate_fn,
)

trainer.train()

results = trainer.evaluate()
print("Evaluation results:", results)
