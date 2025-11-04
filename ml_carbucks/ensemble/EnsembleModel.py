from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from ml_carbucks import DATA_DIR
from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter  # noqa: F401
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter  # noqa: F401
from ml_carbucks.adapters.UltralyticsAdapter import (  # noqa: F401
    YoloUltralyticsAdapter,
    RtdetrUltralyticsAdapter,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import merge_model_predictions
from ml_carbucks.utils.preprocessing import create_clean_loader

logger = setup_logger(__name__)


@dataclass
class EnsembleModel:
    classes: List[str]
    adapters: List[BaseDetectionAdapter]

    def setup(self) -> "EnsembleModel":
        for adapter in self.adapters:
            adapter.setup()
        return self

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
        loader = create_clean_loader(
            datasets, shuffle=False, transforms=None, batch_size=8
        )
        results = []
        for adapter_idx, adapter in enumerate(self.adapters):
            logger.info(f"Evaluating adapter: {adapter.__class__.__name__}")
            for images, targets in tqdm(loader):
                predictions = adapter.predict(images)

                metrics[adapter_idx].update(predictions, targets)  # type: ignore

            metric = metrics[adapter_idx].compute()
            results.append(metric)

        return results

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> Dict[str, float]:

        loader = create_clean_loader(
            datasets, shuffle=False, transforms=None, batch_size=8
        )
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
    classes=["scratch", "dent", "crack"],
    adapters=[
        # YoloUltralyticsAdapter(
        #     classes=["scratch", "dent", "crack"],
        #     **{
        #         "img_size": 384,
        #         "batch_size": 32,
        #         "epochs": 27,
        #         "lr": 0.0015465639515144544,
        #         "momentum": 0.3628781599889685,
        #         "weight_decay": 0.0013127041660177367,
        #         "optimizer": "NAdam",
        #     },
        #     weights="/home/bachelor/ml-carbucks/results/ensemble_demos/trial_4_YoloUltralyticsAdaptermodel.pt",
        # ),
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
        FasterRcnnAdapter(
            classes=["scratch", "dent", "crack"],
            **{
                "img_size": 384,
                "batch_size": 8,
                "epochs": 21,
                "lr_backbone": 2.6373762637681257e-05,
                "lr_head": 0.0011244046084737927,
                "weight_decay_backbone": 0.000796017512818448,
                "weight_decay_head": 0.0005747409908715994,
            },
            weights="/home/bachelor/ml-carbucks/results/ensemble_demos/trial_4_FasterRcnnAdaptermodel.pth",
        ),
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
    ],
).setup()

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
    metrics = ensemble.evaluate_adapters_by_evaluation_from_dataset(val_datasets)  # type: ignore
    for idx, adapter in enumerate(ensemble.adapters):
        logger.info(f"Adapter: {adapter.__class__.__name__}, Metrics: {metrics[idx]}")

    """
    INFO __main__ 17:13:01 | Adapter: YoloUltralyticsAdapter, Metrics: {'map_50': 0.37547850015394535, 'map_50_95': 0.19936537404157367}
    INFO __main__ 17:13:01 | Adapter: RtdetrUltralyticsAdapter, Metrics: {'map_50': 0.49289777293378106, 'map_50_95': 0.27581342862762825}
    INFO __main__ 17:13:01 | Adapter: FasterRcnnAdapter, Metrics: {'map_50': 0.1564657837152481, 'map_50_95': 0.05529939383268356}
    INFO __main__ 17:13:01 | Adapter: EfficientDetAdapter, Metrics: {'map_50': 0.3597193956375122, 'map_50_95': 0.16730839014053345}
    """

    return metrics


def test_2():
    logger.info("Evaluating ensemble by predict from dataset")
    metrics = ensemble.evaluate_adapters_by_predict_from_dataset(val_datasets)  # type: ignore
    for idx, adapter in enumerate(ensemble.adapters):
        logger.info(f"Adapter: {adapter.__class__.__name__}, Metrics: {metrics[idx]}")

    return metrics


def test_3(m1, m2, metric_name: str):
    logger.info("Comparing results from both evaluation methods")
    for idx, adapter in enumerate(ensemble.adapters):
        logger.info(f"Adapter: {adapter.__class__.__name__}")
        logger.info(f"Metrics from evaluation: {m1[idx][metric_name]}")
        logger.info(f"Metrics from predict: {m2[idx][metric_name]}")


if __name__ == "__main__":
    # m1 = test_1()
    m2 = test_2()
    # test_3(m1, m2, "map_50_95")
