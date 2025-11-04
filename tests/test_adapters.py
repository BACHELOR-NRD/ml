from pathlib import Path
from typing import List, Tuple

import pytest

from ml_carbucks import TEST_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.adapters.UltralyticsAdapter import (  # noqa: F401
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter  # noqa: F401
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter  # noqa: F401


@pytest.fixture
def datasets_and_classes() -> Tuple[
    List[Tuple[str | Path, str | Path]],
    List[Tuple[str | Path, str | Path]],
    List[str],
]:

    demo_datasets = [
        (
            TEST_DIR / "mock" / "images" / "demo",
            TEST_DIR / "mock" / "instances_demo_singular_curated.json",
        )
    ]
    demo_classes = ["scratch", "dent"]

    return demo_datasets, demo_datasets, demo_classes  # type: ignore


@pytest.mark.parametrize(
    "adapter_class, params",
    [
        (YoloUltralyticsAdapter, {"epochs": 30, "training_augmentations": False}),
        (RtdetrUltralyticsAdapter, {"epochs": 30, "training_augmentations": False}),
        (FasterRcnnAdapter, {"epochs": 30, "training_augmentations": False}),
        (
            EfficientDetAdapter,
            {
                "epochs": 30,
                "training_augmentations": False,
                "confidence_threshold": 0.01,
            },
        ),
    ],
)
def test_adapter_can_overfit(
    adapter_class: type[BaseDetectionAdapter],
    params: dict,
    datasets_and_classes: Tuple[
        List[Tuple[str | Path, str | Path]],
        List[Tuple[str | Path, str | Path]],
        List[str],
    ],
) -> None:
    """
    This function verifies that the given adapter can overfit on a small dataset.
    It is to make sure that the model and training loop are implemented correctly.
    """

    SCORE_THRESHOLD = 0.5
    train_datasets, val_datasets, classes = datasets_and_classes
    adapter = adapter_class(classes=classes)  # type: ignore

    adapter.set_params(params)
    adapter.setup()

    adapter.fit(train_datasets)
    eval_results = adapter.evaluate(val_datasets)

    print(f"Evaluation results for {adapter_class.__name__}: {eval_results}")
    assert eval_results["map_50_95"] > SCORE_THRESHOLD
