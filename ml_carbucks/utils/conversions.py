import os
import json
import time
import yaml
from pathlib import Path

from PIL import Image

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


def convert_yolo_to_coco(base_dir, splits):

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

    # Convert all splits
    for split in splits:
        images_dir = os.path.join(base_dir, "images", split)
        labels_dir = os.path.join(base_dir, "labels", split)
        output_json = os.path.join(base_dir, f"instances_{split}_curated.json")

        # yolo_to_coco(images_dir, labels_dir, categories, output_json)
        coco = {
            "images": [],
            "annotations": [],
            "categories": categories,
            "info": {
                "description": "",
                "version": "",
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
            label_file = os.path.join(
                labels_dir, os.path.splitext(img_file)[0] + ".txt"
            )
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        cls, x_center, y_center, bw, bh = map(
                            float, line.strip().split()
                        )
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

        logger.info(
            f"Saved {output_json} with {len(coco['images'])} images and {len(coco['annotations'])} annotations."
        )


def convert_coco_to_yolo(img_dir: str, ann_file: str) -> Path:
    start_time = time.time()
    ann_path = Path(ann_file)
    img_dir_path = Path(img_dir)
    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    categories = coco["categories"]

    # === remap class ids to contiguous 0-based indices ===
    id_map = {
        cat["id"]: i for i, cat in enumerate(sorted(categories, key=lambda x: x["id"]))
    }
    names = {i: cat["name"] for cat, i in zip(categories, id_map.values())}

    # === prepare output paths ===
    labels_dir = img_dir_path.parent.parent / "labels" / img_dir_path.name
    labels_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = ann_path.parent / f"{ann_path.stem}.yaml"

    # === group annotations by image ===
    img_to_anns = {}
    for ann in annotations:
        img_id = ann["image_id"]
        img_to_anns.setdefault(img_id, []).append(ann)

    # === write YOLO label files ===
    for img_id, anns in img_to_anns.items():
        img_info = images[img_id]
        w, h = img_info["width"], img_info["height"]
        label_path = labels_dir / (Path(img_info["file_name"]).stem + ".txt")

        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in id_map:
                continue

            bbox = ann["bbox"]  # [x_min, y_min, width, height]
            x_c = (bbox[0] + bbox[2] / 2) / w
            y_c = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h

            lines.append(
                f"{id_map[cat_id]} {round(x_c, 6)} {round(y_c, 6)} {round(bw, 6)} {round(bh, 6)}"
            )

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    # === create YAML file ===
    dataset_yaml = {
        "train": str(Path(img_dir).resolve()),
        "val": str(Path(img_dir).resolve()),
        "nc": len(categories),
        "names": names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    logger.info(f"COCO to YOLO conversion completed in {elapsed_seconds:.2f} seconds")
    if elapsed_seconds > 15:
        logger.warning(
            "COCO to YOLO conversion took longer than expected. "
            "Consider optimizing this process for large datasets."
        )

    return yaml_path
