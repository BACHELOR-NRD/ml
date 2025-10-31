from typing import List
from dataclasses import dataclass

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter
from ml_carbucks.adapters.UltralyticsAdapter import (
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)


@dataclass
class EnsembleModel:
    adapters: List[BaseDetectionAdapter]


ensemble = EnsembleModel(
    adapters=[
        YoloUltralyticsAdapter(
            classes=["scratch", "dent", "crack"],
            weights="/home/bachelor/ml-carbucks/results/optuna/hyper_20251031_155152/trial_0_YoloUltralyticsAdaptermodel.pt",
            img_size=256,
        ),
        RtdetrUltralyticsAdapter(
            classes=["scratch", "dent", "crack"],
            weights="/home/bachelor/ml-carbucks/results/optuna/hyper_20251031_155152/trial_0_RtdetrUltralyticsAdaptermodel.pt",
            img_size=256,
        ),
        FasterRcnnAdapter(
            classes=["scratch", "dent", "crack"],
            weights="/home/bachelor/ml-carbucks/results/optuna/hyper_20251031_155152/trial_0_FasterRcnnAdaptermodel.pth",
            img_size=256,
        ),
        EfficientDetAdapter(
            classes=["scratch", "dent", "crack"],
            weights="/home/bachelor/ml-carbucks/results/optuna/hyper_20251031_155152/trial_0_EfficientDetAdaptermodel.pth",
            img_size=256,
        ),
    ]
)
