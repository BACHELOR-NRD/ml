from ml_carbucks.utils.DatasetsPathManager import DatasetsPathManager
from ml_carbucks.utils.logger import setup_logger
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import shutil
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter
from ml_carbucks.adapters.UltralyticsAdapter import (
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.adapters.BaseDetectionAdapter import (
    BaseDetectionAdapter,
    ADAPTER_DATASETS,
    ADAPTER_METRICS,
)


dataset_base = Path("/home/bachelor/ml-carbucks/data/carbucks_crossval_folds")


def cross_validate(
    adapter: BaseDetectionAdapter,
    train_folds: list[ADAPTER_DATASETS] = DatasetsPathManager.CARBUCKS_TRAIN_CV,
    val_folds: list[ADAPTER_DATASETS] = DatasetsPathManager.CARBUCKS_VAL_CV,
) -> list[ADAPTER_METRICS]:

    fold_metrics: list[ADAPTER_METRICS] = []

    for fold_idx, (train_data, val_data) in enumerate(
        zip(train_folds, val_folds, strict=True)
    ):
        model = adapter.clone()

        model.fit(train_data)

        metrics = model.evaluate(val_data)
        fold_metrics.append(metrics)

    return fold_metrics


def stratified_cross_valitation(
    hyper_results: dict | Path,
    results_dir: Path,
    dataset_dir: Path = dataset_base,
    cv_folds: int = 5,
    setup_ensemble: callable = None,
    ensemble_args: dict = None,
):
    """Perform stratified cross-validation with optional ensemble setup.
    Args:
        hyper_results: Dictionary or path to JSON file containing hyperparameter optimization results.
        results_dir: Directory to save cross-validation results.
        dataset_dir: Directory containing dataset folds.
        cv_folds: Number of cross-validation folds.
        setup_ensemble: Optional callable to set up an ensemble model.
        ensemble_args: Optional dictionary of arguments for the ensemble setup.
    """

    logger = setup_logger(__name__)
    logger.info("Starting stratified cross-validation")

    if isinstance(hyper_results, (Path, str)):
        hyper_results = read_json(Path(hyper_results))

    study_name = hyper_results["study_name"]
    study_dir = get_or_create_study_dir(results_dir, study_name)
    best_params = hyper_results["best_params"]
    folds = sorted(dataset_dir.glob("fold_*"))
    print(len(folds))
    fold_results = []

    for fold in folds:

        fold_idx = int(fold.name.split("_")[1])
        logger.info(f"Processing fold {fold_idx}")

        train_dataset = [(fold / "images" / "train", fold / "annotations_train.json")]
        val_dataset = [(fold / "images" / "val", fold / "annotations_val.json")]

        logger.info(f"Hyperparameters for fold {fold_idx}: {best_params}")
        logger.info(f"training dataset: {train_dataset}")
        logger.info(f"validation dataset: {val_dataset}")

        adapter_class = get_adapter_class(hyper_results["adapter"])
        result = model_training(
            adapter_class,
            best_params,
            train_dataset,
            val_dataset,
            setup_ensemble,
            ensemble_args,
        )
        fold_results.append(result)

    logger.info("Cross-validation complete. Compiling summary statistics.")

    summary = create_summary_statistics(fold_results, study_name, cv_folds)
    summary_path = study_dir / "cv_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary statistics saved to {summary_path}")
    return summary


# helpers


def get_or_create_study_dir(results_dir: Path, study_name: str) -> Path:
    study_dir = results_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir


def model_training(
    adapter_class: type[BaseDetectionAdapter],
    best_params: dict,
    train_dataset: tuple,
    val_dataset: tuple,
    setup_ensemble: callable = None,
    ensemble_args: dict = None,
) -> dict:

    if setup_ensemble:
        return setup_ensemble(train_dataset, val_dataset, best_params, ensemble_args)
    else:
        model: BaseDetectionAdapter = adapter_class(**best_params)
        model.fit(train_dataset)
        metrics = model.evaluate(val_dataset)
        return metrics


def get_adapter_class(adapter_name: str):
    adapter_classes = {
        "EfficientDetAdapter": EfficientDetAdapter,
        "FasterRcnnAdapter": FasterRcnnAdapter,
        "YoloUltralyticsAdapter": YoloUltralyticsAdapter,
        "RtdetrUltralyticsAdapter": RtdetrUltralyticsAdapter,
    }
    return adapter_classes.get(adapter_name)


def create_fold_datasets(
    coco_data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset_dir: Path,
    study_dir: Path,
    fold_idx: int,
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
        ann for ann in coco_data["annotations"] if ann["image_id"] in selected_image_ids
    ]

    return {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_data["categories"],
    }


def create_summary_statistics(
    fold_results: list[dict], study_name: str, cv_folds: int
) -> dict:

    summary = {
        "study_name": study_name,
        "cv_folds": cv_folds,
        "fold_results": fold_results,
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
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    folds = []
    for train_idx, val_idx in mskf.split(X, Y):
        folds.append((train_idx, val_idx))

    return folds


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def create_crossval_fold_structure(
    source_dataset_dir: Path,
    output_base_dir: Path,
    coco_annotations_file: str = "instances_crossval.json",
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """
    Creates fold directories with train/val splits using multilabel stratified k-fold.

    Args:
        source_dataset_dir: Directory containing 'images/' and 'annotations/' folders
        output_base_dir: Base directory where fold1, fold2, etc. will be created
        coco_annotations_file: Name of the COCO annotations JSON file
        n_splits: Number of folds to create
        random_state: Random seed for reproducibility

    Structure created:
        output_base_dir/
            fold_1/
                images/
                    train/
                    val/
                annotations_train.json
                annotations_val.json
            fold_2/
            ...
    """
    logger = setup_logger(__name__)

    annotations_path = source_dataset_dir / coco_annotations_file
    logger.info(f"Loading annotations from {annotations_path}")
    coco_data = read_json(annotations_path)

    # Get image directory
    images_dir = source_dataset_dir / "images"
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Create multilabel stratified k-fold splits
    logger.info(f"Creating {n_splits}-fold stratified split")
    folds = create_multilabel_stratified_kfold_split(coco_data, n_splits, random_state)

    # Create output base directory
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Process each fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        logger.info(f"Processing fold {fold_idx}/{n_splits}")

        # Create fold directory structure
        fold_dir = output_base_dir / f"fold_{fold_idx}"
        base_img_dir = fold_dir / "images"
        train_images_dir = base_img_dir / "train"
        val_images_dir = base_img_dir / "val"

        # Create directories
        train_images_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)

        # Filter annotations for train and val
        train_annotations = filter_coco_by_indices(coco_data, train_idx)
        val_annotations = filter_coco_by_indices(coco_data, val_idx)

        # Save annotations
        train_ann_path = fold_dir / "annotations_train.json"
        val_ann_path = fold_dir / "annotations_val.json"

        with open(train_ann_path, "w") as f:
            json.dump(train_annotations, f, indent=2)
        with open(val_ann_path, "w") as f:
            json.dump(val_annotations, f, indent=2)

        # Copy images for train set
        logger.info(f"  Copying {len(train_annotations['images'])} training images")
        for img_info in train_annotations["images"]:
            src_path = images_dir / img_info["file_name"]
            dst_path = train_images_dir / img_info["file_name"]
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                logger.warning(f"  Image not found: {src_path}")

        # Copy images for val set
        logger.info(f"  Copying {len(val_annotations['images'])} validation images")
        for img_info in val_annotations["images"]:
            src_path = images_dir / img_info["file_name"]
            dst_path = val_images_dir / img_info["file_name"]
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                logger.warning(f"  Image not found: {src_path}")

        logger.info(f"  Fold {fold_idx} complete: {fold_dir}")

    logger.info(f"All {n_splits} folds created successfully in {output_base_dir}")
