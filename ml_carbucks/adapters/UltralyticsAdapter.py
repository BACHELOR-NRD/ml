from pathlib import Path
from typing import Dict, Any, List
from ultralytics.models.yolo import YOLO
from ultralytics.models.rtdetr import RTDETR

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


class UltralyticsAdapter(BaseDetectionAdapter):

    def load_model(self):
        logger.info("Loading Ultralytics model...")

        if self.model_path and self.model_path.exists():
            self._load_existing_model()
        else:
            self._create_model()

        self.model.to(self.device)

    def _create_model(self):
        model_type = self.metadata.get("model_type", None)
        model_version = self.metadata.get("model_version", None)

        if not model_type or not model_version:
            raise ValueError(
                "model_type and model_version must be specified in metadata to create a new model."
            )

        if model_type == "yolo":
            self.model = YOLO(model_version)
            logger.info(f"Created new YOLO model: {model_version}")

        elif model_type == "rtdetr":
            self.model = RTDETR(model_version)
            logger.info(f"Created new RTDETR model: {model_version}")

    def _load_existing_model(self):
        model_type = self.hparams.get("model_type", None)

        if not model_type:
            raise ValueError(
                "model_type must be specified in hparams to load an existing model."
            )

        if model_type == "yolo":
            self.model = YOLO(str(self.model_path))
            logger.info(f"Loaded existing YOLO model from {self.model_path}")

        elif model_type == "rtdetr":
            self.model = RTDETR(str(self.model_path))
            logger.info(f"Loaded existing RTDETR model from {self.model_path}")

    def setup(self):
        self.data_yaml_path = self.datasets.get("data_yaml", None)
        if not self.data_yaml_path:
            raise ValueError(
                "data_yaml path must be provided in datasets to load data."
            )

    def fit(self):
        logger.info("Starting training...")

        seed = self.metadata.get("seed", 42)
        project_dir = self.metadata.get("project_dir", None)
        name = self.metadata.get("run_name", "ultralytics_run")

        self.model.train(
            data=self.data_yaml_path,
            seed=seed,
            name=name,
            val=False,
            verbose=False,
            save=False,
            project=project_dir,
            **self.hparams,
        )

    def evaluate(self) -> Dict[str, float]:
        logger.info("Starting evaluation...")
        results = self.model.val(data=self.data_yaml_path)

        metrics = {
            "map_50": results.results_dict["metrics/mAP50(B)"],
            "map_50_95": results.results_dict["metrics/mAP50-95(B)"],
        }

        return metrics

    def predict(self, images: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError("Predict method is not yet implemented.")

    def save_model(self, save_path: Path | str):
        logger.info(f"Saving model to {save_path}...")
        self.model.save(save_path)
