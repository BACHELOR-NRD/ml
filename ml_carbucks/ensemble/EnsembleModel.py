from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Literal, Optional, override

import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_METRICS,
    ADAPTER_PREDICTION,
    ADAPTER_DATASETS,
    BaseDetectionAdapter,
)
from ml_carbucks.utils.ensemble import (
    ScoreDistribution,
    fuse_adapters_predictions,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import postprocess_evaluation_results
from ml_carbucks.utils.preprocessing import create_clean_loader

logger = setup_logger(__name__)


@dataclass
class EnsembleModel(BaseDetectionAdapter):
    """
    Lightweight coordinator for multiple detection adapters.
    Handles adapter setup, per-adapter evaluation, and fused inference/eval.
    """

    # --- SETUP PARAMETERS ---

    adapters: List[BaseDetectionAdapter] = field(default_factory=list)
    distributions: Optional[List[ScoreDistribution]] = None

    # --- FUSION PARAMETERS ---

    fusion_strategy: Optional[Literal["nms", "wbf"]] = "nms"
    fusion_conf_threshold: float = 0.2
    fusion_iou_threshold: float = 0.5
    fusion_max_detections: int = 10
    fusion_norm_method: Optional[Literal["minmax", "zscore"]] = "minmax"
    fusion_trust_weights: Optional[list[float]] = None

    # --- MAIN METHODS ---

    def setup(self) -> "EnsembleModel":
        for adapter in self.adapters:
            adapter.setup()
        return self

    def fit(self, datasets: ADAPTER_DATASETS) -> "EnsembleModel":
        logger.warning(
            "EnsembleModel.fit() called - fitting individual adapters from empty datasets."
        )
        for i in range(len(self.adapters)):
            self.adapters[i] = (
                self.adapters[i]
                .clone()
                .set_params({"weights": "DEFAULT"})
                .setup()
                .fit(datasets=datasets)
            )

        return self

    def evaluate(
        self,
        datasets: List[Tuple[str | Path, str | Path]],
        batch_size: int = 8,
    ) -> ADAPTER_METRICS:
        """
        Evaluate fused ensemble predictions against the provided datasets.
        """

        loader = create_clean_loader(
            datasets=datasets,
            shuffle=False,
            transforms=None,
            batch_size=batch_size,
        )

        predictions: List[ADAPTER_PREDICTION] = []
        ground_truths: List[dict] = []
        for images, targets in tqdm(
            loader, desc="Ensemble loader", disable=not self.verbose
        ):
            batch_preds = self.predict(images)
            predictions.extend(batch_preds)
            ground_truths.extend(
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                }
                for target in targets
            )

        metric = MeanAveragePrecision()
        metric.update(predictions, ground_truths)  # type: ignore
        processed = postprocess_evaluation_results(metric.compute())
        return processed

    @override
    def predict(  # type: ignore
        self,
        images: List[np.ndarray],
    ) -> List[ADAPTER_PREDICTION]:
        """
        Run all adapters on a batch of images and return fused predictions.
        """
        adapters_predictions = [adapter.predict(images) for adapter in self.adapters]

        if self.fusion_norm_method is not None and self.distributions is None:
            logger.error(
                "Unable to applay normalizaion for the ensemble predictions."
                + "This is beacuse scores distributions will differ from the ones that it trained based on."
            )
            raise Exception(
                "Unable to applay normalization because normalization metadata is not present."
            )

        fused_predictions = fuse_adapters_predictions(
            adapters_predictions=adapters_predictions,
            max_detections=self.fusion_max_detections,
            iou_threshold=self.fusion_iou_threshold,
            conf_threshold=self.fusion_conf_threshold,
            strategy=self.fusion_strategy,
            trust_weights=self.fusion_trust_weights,
            score_normalization_method=self.fusion_norm_method,
            distributions=self.distributions,
        )
        return fused_predictions

    def debug(
        self,
        train_datasets: ADAPTER_DATASETS,
        val_datasets: ADAPTER_DATASETS,
        results_path: str | Path,
        results_name: str,
        visualize: Literal["every", "last", "none"] = "none",
    ) -> ADAPTER_METRICS:
        raise NotImplementedError("EnsembleModel debugging not implemented yet.")

    def save_weights(
        self,
        dir: Path | str,
        prefix: str = "",
        suffix: str = "",
    ) -> Path:
        raise NotImplementedError("EnsembleModel weights saving not implemented yet.")

    def save_pickled(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        raise NotImplementedError("EnsembleModel pickling not implemented yet.")

    @staticmethod
    def load_pickled(path: str | Path) -> BaseDetectionAdapter:
        raise NotImplementedError("EnsembleModel loading not implemented yet.")


class EnsembleFacilitator:
    """
    This is a static class that provides helper methods to evaluate and predict
    using an EnsembleModel and its adapters.
    """

    @staticmethod
    def evaluate_adapters_by_evaluation_from_dataset(
        ensemble: EnsembleModel,
        datasets: List[Tuple[str | Path, str | Path]],
    ) -> List[ADAPTER_METRICS]:
        metrics = []
        adapters: List[BaseDetectionAdapter] = ensemble.adapters
        for adapter in adapters:
            adapter_metrics = adapter.evaluate(datasets)
            metrics.append(adapter_metrics)
        return metrics

    @staticmethod
    def evaluate_adapters_by_predict_from_dataset(
        ensemble: EnsembleModel,
        datasets: List[Tuple[str | Path, str | Path]],
        batch_size: int = 8,
        verbose: bool = True,
    ) -> List[ADAPTER_METRICS]:

        loader = create_clean_loader(
            datasets=datasets,
            shuffle=False,
            transforms=None,
            batch_size=batch_size,
        )

        adapters: List[BaseDetectionAdapter] = ensemble.adapters
        adapters_predictions: List[List[ADAPTER_PREDICTION]] = [[] for _ in adapters]
        ground_truths: List[dict] = []
        for images, targets in tqdm(
            loader, desc="Ensemble loader", disable=not verbose
        ):
            for adapter_idx, adapter in enumerate(adapters):
                preds = adapter.predict(images)
                adapters_predictions[adapter_idx].extend(
                    {
                        "boxes": pred["boxes"].detach().cpu(),
                        "scores": pred["scores"].detach().cpu(),
                        "labels": pred["labels"].detach().cpu(),
                    }
                    for pred in preds
                )

            ground_truths.extend(
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                }
                for target in targets
            )

        final_results: List[ADAPTER_METRICS] = []
        for adapter_idx, adapter in enumerate(adapters):
            logger.info(f"Evaluating adapter predictions: {adapter.__class__.__name__}")
            metric = MeanAveragePrecision()
            metric.update(adapters_predictions[adapter_idx], ground_truths)  # type: ignore
            processed = postprocess_evaluation_results(metric.compute())
            logger.info(
                "%s metrics -> map_50: %.3f | map_75: %.3f | map_50_95: %.3f",
                adapter.__class__.__name__,
                processed["map_50"],
                processed["map_75"],
                processed["map_50_95"],
            )
            final_results.append(processed)
        return final_results

    @staticmethod
    def predict_from_datasets(
        ensemble: EnsembleModel,
        datasets: ADAPTER_DATASETS,
        batch_size: int = 8,
    ) -> List[ADAPTER_PREDICTION]:
        """
        Run ensemble prediction for every sample in the datasets.
        Returns (fused predictions, ground truths, per-adapter predictions).
        """

        loader = create_clean_loader(
            datasets=datasets,
            shuffle=False,
            transforms=None,
            batch_size=batch_size,
        )

        predictions: List[ADAPTER_PREDICTION] = []
        for images, _ in tqdm(
            loader, desc="Ensemble loader", disable=not ensemble.verbose
        ):
            batch_preds = ensemble.predict(images)
            predictions.extend(batch_preds)

        return predictions
