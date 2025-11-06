# ML ComputerVision

End-to-end research sandbox for vehicle damage detection. The repository standardises training, evaluating, and ensembling multiple object-detection backbones so the team can iterate quickly on model quality for the CarBucks bachelor project.

## Table of Contents

-   [Architecture](#architecture)
-   [Workflow](#workflow)
-   [Getting Started](#getting-started)
-   [Highlighted Implementations](#highlighted-implementations)
-   [Datasets](#datasets)
-   [Testing](#testing)
-   [Notebooks & Utilities](#notebooks--utilities)

## Architecture

### Package layout

```
ml_carbucks/
├── adapters/            # Model wrappers that share a common detection API
├── optmization/         # Hyper-parameter search & early-stopping utilities
├── ensemble/            # Model ensembling utilities and evaluation helpers
├── patches/             # Local fixes & extensions for third-party libraries
├── utils/               # Shared preprocessing, postprocessing, logging, IO
├── notebooks/           # Research notebooks for debugging and analysis -> it is not kept up to date
└── milestones/          # Milestones demos and showcases of key experiments
```

Key modules:

-   `adapters.BaseDetectionAdapter`: abstract contract that unifies setup/fit/evaluate/predict/save across backbones.
-   `adapters.EfficientDetAdapter`, `FasterRcnnAdapter`, `UltralyticsAdapter` (+ YOLO & RT-DETR specialisations): concrete implementations that map the contract onto each framework.
-   `ensemble.EnsembleModel`: bootstraps multiple adapters, evaluates them uniformly, and merges per-image predictions.
-   `optmization.hyper` & `EarlyStoppingCallback`: wrap Optuna studies with reusable early stopping and adapter cloning.
-   `optimization.TrialParamWrapper`: assists in mapping Optuna trial parameters onto adapter hyper-parameters.
-   `patches.effdet`: a curated fork of EfficientDet dataset pipeline with bug fixes, dataset concatenation, and full COCO metric reporting.
-   `utils.*`:
    -   `preprocessing` – Albumentations pipeline, COCO loaders, batch creation.
    -   `postprocessing` – class-aware NMS & Weighted Boxes Fusion helpers.
    -   `ensemble` – fusion strategies across adapters.
    -   `inference` – quick visualisation helpers for qualitative review.
    -   `result_saver` & `logger` – CSV logging, plots, and consistent logging configuration.

## Workflow

1. **Data preparation** – Curate COCO-style datasets under `data/`. For each dataset, there should be an `images/<folder_id>` with images and an `instances_<annotation_id>.json` file with annotations. Those path pairs are passed to adapters during training and evaluation. There is also a built-in COCO-to-YOLO converter for Ultralytics-based adapters, so there is **no need** to manually create YOLO label files.
2. **Adapter setup** – Instantiate an adapter with class names, set hyper-parameters, and call `setup()` to load pretrained weights onto the available device.
3. **Training** – Use `fit()` with one or more `(image_dir, coco_annotations.json)` tuples. Adapters share augmentation toggles to ease ablation work.
4. **Evaluation** – Call `evaluate()` or `debug()` to obtain mAP metrics and optional CSV/plot tracking through `ResultSaver`.
5. **Prediction & visual review (optioanl)** – Run `predict()` on raw `numpy` images and visualise with `utils.inference.plot_img_pred*` helpers.
6. **Hyper-parameter search (optional, recommended)** – Launch Optuna studies with `optmization.hyper` to explore adapter parameter spaces, using early stopping for budget control.
7. **Ensembling (optional, recommended)** – Wrap adapters inside `EnsembleModel`, evaluate them jointly, or fuse predictions via NMS/WBF (`utils.ensemble`).

Example (YOLO adapter):

```python
from ml_carbucks.adapters.UltralyticsAdapter import YoloUltralyticsAdapter

classes = ["scratch", "dent", "crack"]
train_data = [("data/car_dd/images/train", "data/car_dd/instances_train_curated.json")]
val_data = [("data/car_dd/images/val", "data/car_dd/instances_val_curated.json")]

adapter = (
		YoloUltralyticsAdapter(classes=classes)
		.set_params({"epochs": 30, "batch_size": 16})
		.setup()
		.fit(train_data)
)

metrics = adapter.evaluate(val_data)
print(metrics)
```

## Getting Started

### Prerequisites

-   Python 3.12–3.13
-   [Poetry](https://python-poetry.org/) for dependency management
-   GPU with CUDA for training (CPU is sufficient for quick smoke tests)

### Installation

Clone the repository and install dependencies:

```bash
poetry install
```

Enter the virtual environment when developing locally:

```bash
poetry shell
```

### Project layout conventions

-   Place curated datasets inside `data/`. The package exposes helpful constants such as `ml_carbucks.DATA_DIR` and `ml_carbucks.DATA_CAR_DD_DIR`.
-   Training artefacts (weights, CSV logs, plots) are saved under `results/` by default.
-   Logging output streams to both stdout and `logs/logs.log` via `utils.logger`.

## Highlighted Implementations

-   **Unified adapter abstraction** – `BaseDetectionAdapter` standardises lifecycle operations and exposes `.clone()` / `.set_params()` helpers, making it trivial to share code between research backbones.
-   **Robust data pipeline** – `utils.preprocessing` wraps the COCO loader with Albumentations-based augmentations, padding, and deterministic label remapping.
-   **Ultralytics interoperability** – `UltralyticsAdapter.coco_to_yolo()` converts COCO targets to YOLO text files and YAML manifests on the fly, streamlining experiments across ecosystems.
-   **Prediction fusion** – `utils.ensemble.fuse_adapters_predictions()` supports score normalisation, per-class NMS, and Weighted Boxes Fusion for ensemble research.
-   **Automated logging** – `ResultSaver` keeps a CSV trace and plots per epoch, while Optuna callbacks checkpoint top-performing models during hyper-parameter sweeps. ResultsSaver is integrated into some adapters that support epoch-level hooks.

## Datasets

It is **important** to mentioned that the datasets should be curated into COCO format with 1-based category IDs before use 0 class label is reserved as background for some of adapetrs.

Primary dataset: **CarDD** – raw for automotive damage detection.

-   Homepage: <https://cardd-ustc.github.io/>
-   Citation: Wang, Li, Wu. _CarDD: A New Dataset for Vision-Based Car Damage Detection_, IEEE T-ITS, 2023. DOI: [10.1109/TITS.2023.3258480](https://doi.org/10.1109/TITS.2023.3258480)

Usage tips:

-   Convert or curate splits into COCO JSON files (`instances_*.json`) and mirror images into `data/car_dd/images/<split>`.
-   For YOLO-based training, the conversion routine (see above) writes labels into `data/car_dd/labels/<split>` automatically.
-   Keep lightweight demo assets under `tests/mock/` for fast regression checks.

## Testing

Run the automated checks from the project root:

```bash
poetry run pytest
```

Test coverage highlights:

-   `tests/test_adapters.py` asserts every adapter can overfit a micro-dataset (GPU recommended). This ensures that training and evaluation actually work as expected.
-   `tests/test_postprocessing.py` validates confidence filtering, IoU suppression, and max-detection limits in `postprocess_prediction_nms`.

## Notebooks & Utilities

-   Research notebooks live under `ml_carbucks/notebooks/` with subfolders for bug-fixing, exploration, and other.
-   Helper scripts in `scripts/` assist with dataset curation (e.g., duplicate detection, label conversion).
-   `docs/duplicate_images.json` collects metadata surfaced during dataset cleaning.

Contributions that extend adapters, ensemble strategies, or dataset tooling are welcome—please keep the README sections above up to date when new workflows land.
