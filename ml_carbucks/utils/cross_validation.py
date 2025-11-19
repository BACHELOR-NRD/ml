from ml_carbucks.utils.logger import setup_logger
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter
from ml_carbucks.adapters.UltralyticsAdapter import UltralyticsAdapter
from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter


def stratified_cross_valitation(hyper_results :dict | Path,
                                results_dir: Path,
                                dataset_dir:Path = 'data/carbucks_crossval_dataset/images',
                                annotations_path:Path = 'data/carbucks_crossval_dataset/annotations/instances_crossval.json',
                                cv_folds: int = 5,
                                random_state: int = 42):
    
    logger = setup_logger(__name__)
    if isinstance(hyper_results, Path):
        hyper_results = read_json(hyper_results)
        
    study_name = hyper_results["study_name"]
    study_dir = results_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    best_params = hyper_results["best_params"]
    coco_data = read_json(annotations_path)
    folds = create_multilabel_stratified_kfold_split(coco_data, cv_folds, random_state)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"Processing fold {fold_idx + 1}/{cv_folds}")
        
        # Create fold-specific datasets
        train_dataset, val_dataset = create_fold_datasets(
            coco_data, train_idx, val_idx, dataset_dir, study_dir, fold_idx
        )

        adapter_class = get_adapter_class(hyper_results["adapter"])
        model: BaseDetectionAdapter = adapter_class(
            classes=hyper_results['classes'],
            img_size=best_params["img_size"],
            batch_size=best_params["batch_size"],
            epochs=best_params["epochs"],
            lr=best_params["lr"],
            momentum=best_params["momentum"],
            weight_decay=best_params["weight_decay"],
            optimizer=best_params["optimizer"],
            project_dir=None,
            training_save=False,
            name=f"fold_{fold_idx}"
        )

        model.setup()
        model.fit(train_dataset)
        metrics = model.evaluate(val_dataset)

        train_path = study_dir / f"fold_{fold_idx}_train.json"
        val_path = study_dir / f"fold_{fold_idx}_val.json"
        train_path.unlink()
        val_path.unlink()

        fold_data = {"fold": fold_idx, "metrics": metrics}
        fold_file = study_dir / f"fold_{fold_idx}.json"
        with open(fold_file, "w") as f:
            json.dump(fold_data, f, indent=4)

        fold_results.append(fold_data)

    logger.info("Cross-validation complete. Compiling summary statistics.")
    summary = create_summary_statistics(fold_results, study_name, cv_folds)
    summary_path = study_dir / "cv_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


#helpers
def get_adapter_class(adapter_name: str):
    adapter_classes = {
        "EfficientDetAdapter": EfficientDetAdapter,
        "FasterRcnnAdapter": FasterRcnnAdapter,
        "UltralyticsAdapter": UltralyticsAdapter
    }
    return adapter_classes.get(adapter_name)

def create_fold_datasets(
    coco_data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset_dir: Path,
    study_dir: Path,
    fold_idx: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:

    train_annotations = filter_coco_by_indices(coco_data, train_idx)
    val_annotations = filter_coco_by_indices(coco_data, val_idx)

    train_path = study_dir / f"fold_{fold_idx}_train.json"
    val_path = study_dir / f"fold_{fold_idx}_val.json"

    with open(train_path, "w") as f:
        json.dump(train_annotations, f)
    with open(val_path, "w") as f:
        json.dump(val_annotations, f)

    train_dataset = [(dataset_dir, train_path)]
    val_dataset = [(dataset_dir, val_path)]
    
    return train_dataset, val_dataset

def filter_coco_by_indices(coco_data: dict, image_indices: np.ndarray) -> dict:
    images = coco_data["images"]
    selected_image_ids = {images[idx]["id"] for idx in image_indices}
    
    filtered_images = [img for img in images if img["id"] in selected_image_ids]
    filtered_annotations = [
        ann for ann in coco_data["annotations"] 
        if ann["image_id"] in selected_image_ids
    ]
    
    return {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_data["categories"]
    }


def create_summary_statistics(fold_results: list[dict], study_name: str, cv_folds: int) -> dict:

    summary = {
        "study_name": study_name,
        "cv_folds": cv_folds,
        "fold_results": fold_results,
        "avg_map_50": float(np.mean([f["metrics"]["map_50"] for f in fold_results])),
        "avg_map_50_95": float(np.mean([f["metrics"]["map_50_95"] for f in fold_results])),
    }
    return summary



def coco_to_multilabel_matrix(coco_data: dict):
    """
    Returns:
        Y: multi-label matrix [num_images × num_classes]
    """

    annotations = coco_data["annotations"]
    images = coco_data["images"]

    category_ids = sorted({ann["category_id"] for ann in annotations})
    num_classes = len(category_ids)
    cat_to_index = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    img_to_categories = defaultdict(list)
    for ann in annotations:
        img_to_categories[ann["image_id"]].append(ann["category_id"])

    image_ids = [img["id"] for img in images]
    Y = np.zeros((len(image_ids), num_classes), dtype=int)

    for i, img_id in enumerate(image_ids):
        for cat_id in img_to_categories.get(img_id, []):
            col = cat_to_index[cat_id]
            Y[i, col] = 1


    return Y

def create_multilabel_stratified_kfold_split(coco_data, n_splits=5, random_state=42):
    
    Y = coco_to_multilabel_matrix(coco_data)
   
    # Dummy feature matrix
    X = np.zeros((len(Y), 1))

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    folds = []
    for train_idx, val_idx in mskf.split(X, Y):
        folds.append((train_idx, val_idx))

    return folds


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    

