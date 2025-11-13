from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Literal, Optional

import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_METRICS,
    ADAPTER_PREDICTION,
    BaseDetectionAdapter,
)
from ml_carbucks.ensemble.merging import fuse_adapters_predictions
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import postprocess_evaluation_results
from ml_carbucks.utils.preprocessing import create_clean_loader

logger = setup_logger(__name__)


@dataclass
class EnsembleModel:
    """
    Lightweight coordinator for multiple detection adapters.
    Handles adapter setup, per-adapter evaluation, and fused inference/eval.
    """

    classes: List[str]
    adapters: List[BaseDetectionAdapter]
    fusion_strategy: Optional[Literal["nms", "wbf"]] = "nms"
    fusion_conf_threshold: float = 0.25
    fusion_iou_threshold: float = 0.55
    fusion_max_detections: int = 300
    loader_batch_size: int = 8
    loader_shuffle: bool = False
    loader_transforms: Optional[object] = field(default=None, repr=False)
    fusion_apply_normalization: bool = False
    fusion_norm_method: Literal["minmax", "zscore"] = "minmax"
    fusion_trust_weights: Optional[list[float]] = None


    def setup(self) -> "EnsembleModel":
        for adapter in self.adapters:
            adapter.setup()
        return self

    @staticmethod
    def _prediction_to_cpu(pred: ADAPTER_PREDICTION) -> ADAPTER_PREDICTION:
        """Detach adapter outputs so downstream metrics stay on CPU."""
        return {
            "boxes": pred["boxes"].detach().cpu(),
            "scores": pred["scores"].detach().cpu(),
            "labels": pred["labels"].detach().cpu(),
        }

    def _collect_adapter_predictions(
        self,
        datasets: List[Tuple[str | Path, str | Path]],
        *,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> Tuple[List[List[ADAPTER_PREDICTION]], List[dict]]:
        """
        Run every adapter across the provided datasets and collect predictions.
        Returns per-adapter prediction lists plus ground-truth annotations.
        """
        loader = create_clean_loader(
            datasets=datasets,
            shuffle=self.loader_shuffle if shuffle is None else shuffle,
            transforms=self.loader_transforms,
            batch_size=self.loader_batch_size if batch_size is None else batch_size,
        )

        adapters_predictions: List[List[ADAPTER_PREDICTION]] = [
            [] for _ in self.adapters
        ]
        ground_truths: List[dict] = []

        logger.info("Collecting adapter predictions...")
        for images, targets in tqdm(loader, desc="Ensemble loader"):
            for adapter_idx, adapter in enumerate(self.adapters):
                preds = adapter.predict(images)
                adapters_predictions[adapter_idx].extend(
                    self._prediction_to_cpu(pred) for pred in preds
                )

            ground_truths.extend(
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                }
                for target in targets
            )

        logger.info("Finished collecting adapter predictions.")
        return adapters_predictions, ground_truths

    def _resolve_fusion_params(
        self,
        strategy: Optional[Literal["nms", "wbf"]],
        conf_threshold: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
    ) -> Tuple[Optional[Literal["nms", "wbf"]], float, float, int]:
        resolved_strategy = strategy if strategy is not None else self.fusion_strategy
        resolved_conf = (
            conf_threshold if conf_threshold is not None else self.fusion_conf_threshold
        )
        resolved_iou = (
            iou_threshold if iou_threshold is not None else self.fusion_iou_threshold
        )
        resolved_max = (
            max_detections
            if max_detections is not None
            else self.fusion_max_detections
        )
        return resolved_strategy, resolved_conf, resolved_iou, resolved_max

    def evaluate_adapters_by_evaluation_from_dataset(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> List[dict]:
        metrics = []
        for adapter in self.adapters:
            adapter_metrics = adapter.evaluate(datasets)
            metrics.append(adapter_metrics)
        return metrics

    def evaluate_adapters_by_predict_from_dataset(
        self,
        datasets: List[Tuple[str | Path, str | Path]],
        *,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> List[ADAPTER_METRICS]:
        adapters_predictions, ground_truths = self._collect_adapter_predictions(
            datasets=datasets,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        final_results: List[ADAPTER_METRICS] = []
        for adapter_idx, adapter in enumerate(self.adapters):
            logger.info(f"Evaluating adapter predictions: {adapter.__class__.__name__}")
            metric = MeanAveragePrecision()
            metric.update(adapters_predictions[adapter_idx], ground_truths)  # type: ignore
            processed = postprocess_evaluation_results(metric.compute())
            logger.info(
                "%s metrics -> map_50: %.3f | map_75: %.3f | map_50_95: %.3f",
                adapter.__class__.__name__,
                processed["map_50"],
                processed["map_75"],
                processed["map_50_95"]
            )
            final_results.append(processed)
        return final_results

    def predict(
        self,
        images: List[np.ndarray],
        *,
        strategy: Optional[Literal["nms", "wbf"]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> List[ADAPTER_PREDICTION]:
        """
        Run all adapters on a batch of images and return fused predictions.
        """
        adapters_predictions = [
            adapter.predict(images) for adapter in self.adapters
        ]
        resolved_strategy, resolved_conf, resolved_iou, resolved_max = (
            self._resolve_fusion_params(
                strategy=strategy,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )
        )

        fused_predictions = fuse_adapters_predictions(
            adapters_predictions=adapters_predictions,
            max_detections=resolved_max,
            iou_threshold=resolved_iou,
            conf_threshold=resolved_conf,
            strategy=resolved_strategy,
            apply_score_normalization=self.fusion_apply_normalization,
            trust_weights=self.fusion_trust_weights,
            score_normalization_method=self.fusion_norm_method,
        )
        return fused_predictions

    def predict_from_datasets(
        self,
        datasets: List[Tuple[str | Path, str | Path]],
        *,
        strategy: Optional[Literal["nms", "wbf"]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> Tuple[
        List[ADAPTER_PREDICTION],
        List[dict],
        List[List[ADAPTER_PREDICTION]],
    ]:
        """
        Run ensemble prediction for every sample in the datasets.
        Returns (fused predictions, ground truths, per-adapter predictions).
        """
        adapters_predictions, ground_truths = self._collect_adapter_predictions(
            datasets=datasets,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        resolved_strategy, resolved_conf, resolved_iou, resolved_max = (
            self._resolve_fusion_params(
                strategy=strategy,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )
        )

        fused_predictions = fuse_adapters_predictions(
            adapters_predictions=adapters_predictions,
            max_detections=resolved_max,
            iou_threshold=resolved_iou,
            conf_threshold=resolved_conf,
            strategy=resolved_strategy,
            apply_score_normalization=self.fusion_apply_normalization,
            trust_weights=self.fusion_trust_weights,
            score_normalization_method=self.fusion_norm_method,
        )
        return fused_predictions, ground_truths, adapters_predictions

    def evaluate(
        self,
        datasets: List[Tuple[str | Path, str | Path]],
        *,
        strategy: Optional[Literal["nms", "wbf"]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> ADAPTER_METRICS:
        """
        Evaluate fused ensemble predictions against the provided datasets.
        """
        fused_predictions, ground_truths, _ = self.predict_from_datasets(
            datasets=datasets,
            strategy=strategy,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        metric = MeanAveragePrecision()
        metric.update(fused_predictions, ground_truths)  # type: ignore
        return postprocess_evaluation_results(metric.compute())
