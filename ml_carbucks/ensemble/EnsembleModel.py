from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional, Type, override

import numpy as np
import pickle as pkl
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_METRICS,
    ADAPTER_PREDICTION,
    ADAPTER_DATASETS,
    BaseDetectionAdapter,
)
from ml_carbucks.adapters.EfficientDetAdapter import EfficientDetAdapter
from ml_carbucks.adapters.FasterRcnnAdapter import FasterRcnnAdapter
from ml_carbucks.adapters.UltralyticsAdapter import (
    RtdetrUltralyticsAdapter,
    YoloUltralyticsAdapter,
)
from ml_carbucks.utils.ensemble_merging import (
    ScoreDistribution,
    fuse_adapters_predictions,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import (
    postprocess_evaluation_results,
    convert_pred2eval,
)
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
    fusion_norm_method: Optional[Literal["minmax", "zscore", "quantile"]] = None
    fusion_trust_factors: Optional[list[float]] = None
    fusion_exponent_factors: Optional[list[float]] = None

    # --- MAIN METHODS ---

    def _setup(self) -> "EnsembleModel":

        if self.checkpoint is not None:
            self._load_from_checkpoint(self.checkpoint)

        elif len(self.adapters) > 0:
            for adapter in self.adapters:
                adapter._setup()
        else:
            raise ValueError("EnsembleModel requires at least one adapter to setup.")

        return self

    def fit(self, datasets: ADAPTER_DATASETS) -> "EnsembleModel":
        for i in range(len(self.adapters)):
            logger.info(
                f"Fitting adapter {i + 1}/{len(self.adapters)} - {self.adapters[i].__class__.__name__}"
            )
            self.adapters[i] = self.adapters[i].fit(datasets=datasets)

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
            predictions.extend([convert_pred2eval(pred) for pred in batch_preds])
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
            trust_factors=self.fusion_trust_factors,
            exponent_factors=self.fusion_exponent_factors,
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

    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        obj = {
            "class_data": {
                "name": self.__class__.__name__,
                "module": self.__class__.__module__,
                "class_type": self.__class__,
            },
            "models": [],
            "params": self.get_params(skip=["adapters", "checkpoint"]),
        }
        pickled_adapter_paths = []
        for idx, adapter in enumerate(self.adapters):
            apath = adapter.save(
                dir=dir, prefix=f"adapter_{idx}_{prefix}", suffix=suffix
            )
            pickled_adapter_paths.append(apath)

        for ppath in pickled_adapter_paths:
            adapter_pickle_dict = pkl.load(open(ppath, "rb"))
            obj["models"].append(adapter_pickle_dict)

        save_path = Path(dir) / f"{prefix}ensemble_model{suffix}.pkl"
        pkl.dump(obj, open(save_path, "wb"))

        return save_path

    def _load_from_checkpoint(
        self, checkpoint_path: str | Path | dict, **kwargs
    ) -> None:
        if isinstance(checkpoint_path, dict):
            obj = checkpoint_path  # type: ignore
        else:
            obj = pkl.load(open(checkpoint_path, "rb"))

        obj_class_name = obj["class_data"]["name"]
        if obj_class_name != self.__class__.__name__:
            raise ValueError(
                f"Pickled adapter class mismatch: expected '{self.__class__.__name__}', got '{obj_class_name}'"
            )

        params = obj["params"]

        # NOTE: perhaps temporary solution that should be improved
        # perhaps could be replaced with adapter_dict["class_data"]["class_type"]
        adaptername_to_class: Dict[str, Type[BaseDetectionAdapter]] = {
            "FasterRcnnAdapter": FasterRcnnAdapter,
            "YoloUltralyticsAdapter": YoloUltralyticsAdapter,
            "RtdetrUltralyticsAdapter": RtdetrUltralyticsAdapter,
            "EfficientDetAdapter": EfficientDetAdapter,
        }

        for adapter_dict in obj["models"]:
            adapter_class_name = adapter_dict["class_data"]["name"]
            if adapter_class_name not in adaptername_to_class:
                raise ValueError(
                    f"Unknown adapter class name '{adapter_class_name}' in checkpoint."
                )
            adapter_class = adaptername_to_class[adapter_class_name]
            adapter = adapter_class(checkpoint=adapter_dict)
            self.adapters.append(adapter)

        self.set_params(params)

    def clone(self) -> "EnsembleModel":
        cloned_adapters = [adapter.clone() for adapter in self.adapters]
        cloned_params = self.get_params(skip=["adapters"])
        cloned_ensemble = EnsembleModel(
            adapters=cloned_adapters,
            **cloned_params,
        )
        return cloned_ensemble


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
