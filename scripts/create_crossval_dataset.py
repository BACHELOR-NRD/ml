import json
import shutil
from pathlib import Path
import argparse
from ml_carbucks.utils.logger import setup_logger



def merge_coco_annotations(file1_path, file2_path, output_path):
    """
    Merge two COCO format annotation files.
    
    Args:
        file1_path: Path to first JSON file (e.g., instances_train_curated.json)
        file2_path: Path to second JSON file (e.g., instances_val_curated.json)
        output_path: Path for merged output file
    """
    
    # Load both JSON files
    with open(file1_path, 'r') as f:
        data1 = json.load(f)
    
    with open(file2_path, 'r') as f:
        data2 = json.load(f)
    
    # Start with data1 as base
    merged_data = {
        "info": data1.get("info", {}),
        "licenses": data1.get("licenses", []),
        "categories": data1.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # Track maximum IDs from file1
    max_image_id = max([img['id'] for img in data1.get('images', [])], default=0)
    max_annotation_id = max([ann['id'] for ann in data1.get('annotations', [])], default=0)
    
    # Add all images from file1
    merged_data['images'].extend(data1.get('images', []))
    
    # Add all annotations from file1
    merged_data['annotations'].extend(data1.get('annotations', []))
    
    # Create mapping for image IDs from file2
    image_id_mapping = {}
    
    # Add images from file2 with new IDs
    for img in data2.get('images', []):
        max_image_id += 1
        old_id = img['id']
        new_img = img.copy()
        new_img['id'] = max_image_id
        image_id_mapping[old_id] = max_image_id
        merged_data['images'].append(new_img)
    
    # Add annotations from file2 with updated IDs
    for ann in data2.get('annotations', []):
        max_annotation_id += 1
        new_ann = ann.copy()
        new_ann['id'] = max_annotation_id
        new_ann['image_id'] = image_id_mapping[ann['image_id']]
        merged_data['annotations'].append(new_ann)
    
    # Save merged data
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    # Print summary
    print(f"Merge completed successfully!")
    print(f"File 1: {len(data1.get('images', []))} images, {len(data1.get('annotations', []))} annotations")
    print(f"File 2: {len(data2.get('images', []))} images, {len(data2.get('annotations', []))} annotations")
    print(f"Merged: {len(merged_data['images'])} images, {len(merged_data['annotations'])} annotations")
    print(f"Output saved to: {output_path}")





def create_merged_dataset(base_path):
    """
    Merge train_curated and val_curated datasets into a single dataset for cross-validation.
    
    Args:
        base_path: Path to the carbucks dataset (data/carbucks)
    """
    logger = setup_logger(__name__)
    base_path = Path(base_path)
    
    # Paths
    train_images_dir = base_path / "images" / "train"
    val_images_dir = base_path / "images" / "val"
    
    
    train_annotations_file = base_path / "instances_train_curated.json"
    val_annotations_file = base_path / "instances_val_curated.json"
    
    # Create merged dataset directory
    merged_dir = Path("data") / "carbucks_crossval_dataset"
    merged_images_dir = merged_dir / "images"
    merged_annotations_dir = merged_dir / "annotations"
    
    merged_images_dir.mkdir(parents=True, exist_ok=True)
    merged_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading annotations...")
    # Load annotations
    with open(train_annotations_file, 'r') as f:
        train_data = json.load(f)
    
    with open(val_annotations_file, 'r') as f:
        val_data = json.load(f)
    
    logger.info(f"Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    logger.info(f"Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    
    # Copy train images
    logger.info("Copying train images...")
    train_copied = 0
    train_missing = 0
    for img in train_data['images']:
        src = train_images_dir / img['file_name']
        dst = merged_images_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)
            train_copied += 1
        else:
            train_missing += 1
            logger.warning(f"Missing train image: {src}")
    
    logger.info(f"Copied {train_copied} train images, {train_missing} missing")
    
    # Copy val images
    logger.info("Copying val images...")
    val_copied = 0
    val_missing = 0
    for img in val_data['images']:
        src = val_images_dir / img['file_name']
        dst = merged_images_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)
            val_copied += 1
        else:
            val_missing += 1
            logger.warning(f"Missing val image: {src}")
    
    logger.info(f"Copied {val_copied} val images, {val_missing} missing")
    
    # Merge annotations with proper ID remapping
    logger.info("Merging annotations...")
    
    # Track maximum IDs from train data
    max_image_id = max([img['id'] for img in train_data.get('images', [])], default=0)
    max_annotation_id = max([ann['id'] for ann in train_data.get('annotations', [])], default=0)
    
    # Start with train data
    merged_images = train_data['images'].copy()
    merged_annotations = train_data['annotations'].copy()
    
    # Create mapping for image IDs from val data
    image_id_mapping = {}
    
    # Add images from val with new IDs
    for img in val_data.get('images', []):
        max_image_id += 1
        old_id = img['id']
        new_img = img.copy()
        new_img['id'] = max_image_id
        image_id_mapping[old_id] = max_image_id
        merged_images.append(new_img)
    
    # Add annotations from val with updated IDs
    for ann in val_data.get('annotations', []):
        max_annotation_id += 1
        new_ann = ann.copy()
        new_ann['id'] = max_annotation_id
        new_ann['image_id'] = image_id_mapping[ann['image_id']]
        merged_annotations.append(new_ann)
    
    merged_data = {
        'info': train_data.get('info', {}),
        'licenses': train_data.get('licenses', []),
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': train_data['categories']
    }
    
    # Save merged annotations
    merged_annotations_file = merged_annotations_dir / "instances_crossval.json"
    with open(merged_annotations_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    logger.info(f"\nMerged dataset created at: {merged_dir}")
    logger.info(f"Total images: {len(merged_data['images'])}")
    logger.info(f"Total annotations: {len(merged_data['annotations'])}")
    logger.info(f"Total images copied: {train_copied + val_copied}")
    logger.info(f"Images directory: {merged_images_dir}")
    logger.info(f"Annotations file: {merged_annotations_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create merged dataset for cross-validation")
    parser.add_argument(
        "--base_path",
        type=str,
        default="data/carbucks",
        help="Path to the carbucks dataset"
    )
    
    args = parser.parse_args()
    create_merged_dataset(args.base_path)