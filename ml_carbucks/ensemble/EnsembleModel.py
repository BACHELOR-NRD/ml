from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.preprocessing import create_clean_loader
from ml_carbucks.utils.postprocessing import process_evaluation_results
from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter

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

        final_results = [process_evaluation_results(metric) for metric in results]
        return final_results
