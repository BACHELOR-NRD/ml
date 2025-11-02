from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ml_carbucks import DATA_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter
from ml_carbucks.adapters.UltralyticsAdapter import (
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import merge_model_predictions
from ml_carbucks.utils.preprocessing import create_loader

logger = setup_logger(__name__)


@dataclass
class EnsembleModel:
    adapters: List[BaseDetectionAdapter]

    def __post_init__(self):
        for adapter in self.adapters:
            adapter.setup()

    def evaluate_adapters_by_evaluation_from_dataset(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> List[dict]:
        metrics = []
        for adapter in self.adapters:
            adapter_metrics = adapter.evaluate(datasets)
            metrics.append(adapter_metrics)
        return metrics

    def evaluate_adapters_by_predict_from_dataset(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> List[dict]:

        metrics = [MeanAveragePrecision() for _ in self.adapters]
        loader = create_loader(datasets, shuffle=False, transforms=None, batch_size=8)
        results = []
        for adapter_idx, adapter in enumerate(self.adapters):
            logger.info(f"Evaluating adapter: {adapter.__class__.__name__}")
            for images, targets in loader:
                predictions = adapter.predict(images)
                metrics[adapter_idx].update(predictions, targets)  # type: ignore

            metric = metrics[adapter_idx].compute()
            results.append(metric)

        return results

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> Dict[str, float]:

        loader = create_loader(datasets, shuffle=False, transforms=None, batch_size=8)
        metric = MeanAveragePrecision()
        for images, targets in loader:
            batch_preds = []
            for adapter in self.adapters:
                preds = adapter.predict(images)
                batch_preds.append(preds)

            merged_batch_preds = merge_model_predictions(
                batch_preds, strategy="wbf", iou_thresh=0.5
            )
            metric.update(merged_batch_preds, targets)  # type: ignore
        final_raw_metric = metric.compute()
        final_curated_metric = {
            "map_50": final_raw_metric["map_50"].item(),
            "map_50_95": final_raw_metric["map_50_95"].item(),
        }
        return final_curated_metric


ensemble = EnsembleModel(
    adapters=[
        YoloUltralyticsAdapter(
            classes=["scratch", "dent", "crack"],
            **{
                "img_size": 384,
                "batch_size": 32,
                "epochs": 27,
                "lr": 0.0015465639515144544,
                "momentum": 0.3628781599889685,
                "weight_decay": 0.0013127041660177367,
                "optimizer": "NAdam",
            },
            weights="/home/bachelor/ml-carbucks/results/ensemble_demos/trial_4_YoloUltralyticsAdaptermodel.pt",
        ),
        # RtdetrUltralyticsAdapter(
        #     classes=["scratch", "dent", "crack"],
        #     **{
        #         "img_size": 384,
        #         "batch_size": 16,
        #         "epochs": 10,
        #         "lr": 0.0001141043015859849,
        #         "momentum": 0.424704619626319,
        #         "weight_decay": 0.00012292547851740234,
        #         "optimizer": "AdamW",
        #     },
        #     weights="/home/bachelor/ml-carbucks/results/ensemble_demos/trial_4_RtdetrUltralyticsAdaptermodel.pt",
        # ),
        # FasterRcnnAdapter(
        #     classes=["scratch", "dent", "crack"],
        #     **{
        #         "img_size": 384,
        #         "batch_size": 8,
        #         "epochs": 21,
        #         "lr_backbone": 2.6373762637681257e-05,
        #         "lr_head": 0.0011244046084737927,
        #         "weight_decay_backbone": 0.000796017512818448,
        #         "weight_decay_head": 0.0005747409908715994,
        #     },
        #     weights="/home/bachelor/ml-carbucks/results/ensemble_demos/trial_4_FasterRcnnAdaptermodel.pth",
        # ),
        # EfficientDetAdapter(
        #     classes=["scratch", "dent", "crack"],
        #     **{
        #         "img_size": 384,
        #         "batch_size": 8,
        #         "epochs": 26,
        #         "optimizer": "momentum",
        #         "lr": 0.003459928723120903,
        #         "weight_decay": 0.0001302610542371722,
        #     },
        #     weights="/home/bachelor/ml-carbucks/results/ensemble_demos/trial_4_EfficientDetAdaptermodel.pth",
        # ),
    ]
)

train_datasets = [
    (
        DATA_DIR / "car_dd_testing" / "images" / "train",
        DATA_DIR / "car_dd_testing" / "instances_train_curated.json",
    )
]
val_datasets = [
    (
        DATA_DIR / "car_dd_testing" / "images" / "val",
        DATA_DIR / "car_dd_testing" / "instances_val_curated.json",
    )
]


def test_1():
    logger.info("Evaluating ensemble by evaluation from dataset")
    metrics = ensemble.evaluate_adapters_by_evaluation_from_dataset(train_datasets)  # type: ignore
    for idx, adapter in enumerate(ensemble.adapters):
        logger.info(f"Adapter: {adapter.__class__.__name__}, Metrics: {metrics[idx]}")

    return metrics


def test_2():
    logger.info("Evaluating ensemble by predict from dataset")
    metrics = ensemble.evaluate_adapters_by_predict_from_dataset(train_datasets)  # type: ignore
    for idx, adapter in enumerate(ensemble.adapters):
        logger.info(f"Adapter: {adapter.__class__.__name__}, Metrics: {metrics[idx]}")

    return metrics


def test_3(m1, m2, metric_name: str):
    logger.info("Comparing results from both evaluation methods")
    for idx, adapter in enumerate(ensemble.adapters):
        logger.info(f"Adapter: {adapter.__class__.__name__}")
        logger.info(f"Metrics from evaluation: {m1[idx][metric_name]}")
        logger.info(f"Metrics from predict: {m2[idx][metric_name]}")


def debug_1():
    loader = create_loader(train_datasets, shuffle=False, transforms=None, batch_size=2)  # type: ignore
    ymodel = YoloUltralyticsAdapter(
        classes=["scratch", "dent", "crack"],
        **{
            "img_size": 384,
            "batch_size": 32,
            "epochs": 27,
            "lr": 0.0015465639515144544,
            "momentum": 0.3628781599889685,
            "weight_decay": 0.0013127041660177367,
            "optimizer": "NAdam",
        },
        weights="/home/bachelor/ml-carbucks/results/ensemble_demos/trial_4_YoloUltralyticsAdaptermodel.pt",
    ).setup()

    for images, targets in loader:
        preds = ymodel.predict(images)
        logger.info(f"Predictions: {preds}")
        logger.info(f"Targets: {targets}")


if __name__ == "__main__":
    debug_1()
    # m1 = test_1()
    # m2 = test_2()
    # test_3(m1, m2, "map_50_95")
