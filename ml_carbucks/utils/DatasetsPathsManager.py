from typing import List, cast

from ml_carbucks import DATA_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import ADAPTER_DATASETS


class DatasetsPathsManager:

    # --- ALL DATA ---

    CARBUCKS_TRAIN_ALL: ADAPTER_DATASETS = cast(
        ADAPTER_DATASETS,
        [
            DATA_DIR / "final_carbucks" / "all" / "images" / "all",
            DATA_DIR / "final_carbucks" / "all" / "instances_all_curated.json",
        ],
    )
    """Datasets used for final training on all CarBucks data.
    Consists of commonly recognized train,val,test split."""

    # --- STANDARD DATA ---

    CARBUCKS_TRAIN_STANDARD: ADAPTER_DATASETS = cast(
        ADAPTER_DATASETS,
        [
            DATA_DIR / "final_carbucks" / "standard" / "images" / "train",
            DATA_DIR / "final_carbucks" / "standard" / "instances_train_curated.json",
        ],
    )
    """Datasets used for simple worklows training development on standard CarBucks data."""

    CARBUCKS_VAL_STANDARD: ADAPTER_DATASETS = cast(
        ADAPTER_DATASETS,
        [
            DATA_DIR / "final_carbucks" / "standard" / "images" / "val",
            DATA_DIR / "final_carbucks" / "standard" / "instances_val_curated.json",
        ],
    )
    """Datasets used for simple worklows validation development on standard CarBucks data."""

    CARBUCKS_TEST_STANDARD: ADAPTER_DATASETS = cast(
        ADAPTER_DATASETS,
        [
            DATA_DIR / "final_carbucks" / "standard" / "images" / "test",
            DATA_DIR / "final_carbucks" / "standard" / "instances_test_curated.json",
        ],
    )
    """Datasets used for simple worklows testing development on standard CarBucks data."""

    # --- CROSS VALIDATION DATA ---
    CARBUCKS_TRAIN_CV: List[ADAPTER_DATASETS] = [
        cast(
            ADAPTER_DATASETS,
            [
                DATA_DIR
                / "final_carbucks"
                / "crossval"
                / f"fold_{i}"
                / "images"
                / "train",
                DATA_DIR
                / "final_carbucks"
                / "crossval"
                / f"fold_{i}"
                / "annotations_train.json",
            ],
        )
        for i in range(5)
    ]
    """Datasets used for cross-validation training on standard CarBucks data."""

    CARBUCKS_VAL_CV: List[ADAPTER_DATASETS] = [
        cast(
            ADAPTER_DATASETS,
            [
                DATA_DIR
                / "final_carbucks"
                / "crossval"
                / f"fold_{i}"
                / "images"
                / "val",
                DATA_DIR
                / "final_carbucks"
                / "crossval"
                / f"fold_{i}"
                / "annotations_val.json",
            ],
        )
        for i in range(5)
    ]
    """Datasets used for cross-validation validation on standard CarBucks data."""

    # --- CARDD DATA ---

    CARDD_TRAIN: ADAPTER_DATASETS = cast(
        ADAPTER_DATASETS,
        [
            DATA_DIR / "car_dd" / "images" / "train",
            DATA_DIR / "car_dd" / "annotations" / "instances_train_curated.json",
        ],
    )
    """Dataset used for training on CarDD dataset."""

    CARDD_VAL: ADAPTER_DATASETS = cast(
        ADAPTER_DATASETS,
        [
            DATA_DIR / "car_dd" / "images" / "val",
            DATA_DIR / "car_dd" / "annotations" / "instances_val_curated.json",
        ],
    )
    """Dataset used for evaluation on CarDD dataset."""

    CARDD_TEST: ADAPTER_DATASETS = cast(
        ADAPTER_DATASETS,
        [
            DATA_DIR / "car_dd" / "images" / "test",
            DATA_DIR / "car_dd" / "annotations" / "instances_test_curated.json",
        ],
    )
    """Dataset used for testing on CarDD dataset."""
