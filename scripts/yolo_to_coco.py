import os
import json
import yaml
from PIL import Image

# Base paths
base_dir = "/home/bachelor/ml-carbucks/data/carbucks"
splits = ["train", "val"]

# Load class names from dataset.yaml
yaml_file = os.path.join(base_dir, "dataset.yaml")
with open(yaml_file, "r") as f:
    data_yaml = yaml.safe_load(f)

# "names" can be dict or list in YOLO .yaml
if isinstance(data_yaml["names"], dict):
    names = [data_yaml["names"][k] for k in sorted(data_yaml["names"].keys())]
else:
    names = data_yaml["names"]

categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]


def yolo_to_coco(images_dir, labels_dir, output_json):
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "info": {
            "description": "Carbucks Dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
    }
    ann_id = 1
    img_id = 1

    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # Image info
        img_path = os.path.join(images_dir, img_file)
        w, h = Image.open(img_path).size

        coco["images"].append(
            {"id": img_id, "file_name": img_file, "width": w, "height": h}
        )

        # Label file
        label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    cls, x_center, y_center, bw, bh = map(float, line.strip().split())
                    x_min = (x_center - bw / 2) * w
                    y_min = (y_center - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h

                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(cls) + 1,
                            "bbox": [x_min, y_min, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(
        f"Saved {output_json} with {len(coco['images'])} images and {len(coco['annotations'])} annotations."
    )


# Convert all splits
for split in splits:
    images_dir = os.path.join(base_dir, "images", split)
    labels_dir = os.path.join(base_dir, "labels", split)
    output_json = os.path.join(base_dir, f"instances_{split}_curated.json")

    yolo_to_coco(images_dir, labels_dir, output_json)
