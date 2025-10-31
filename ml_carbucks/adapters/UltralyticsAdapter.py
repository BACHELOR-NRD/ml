from pathlib import Path
from typing import Dict, Any, List
from ultralytics.models.yolo import YOLO
from ultralytics.models.rtdetr import RTDETR

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


class UltralyticsAdapter(BaseDetectionAdapter):

    def get_possible_hyper_keys(self) -> List[str]:
        # NOTE: This is not an exhaustive list of all possible hyperparameters.
        return [
            "imgsz",
            "optimizer",
            "epochs",
            "batch",
            "lr0",
            "momentum",
            "weight_decay",
            "patience",
        ]

    def get_required_metadata_keys(self) -> List[str]:
        return ["data_yaml", "weights"]

    def fit(self) -> "UltralyticsAdapter":
        logger.info("Starting training...")

        seed = self.get_metadata_value("seed", 42)
        project_dir = self.get_metadata_value("project_dir", None)
        save = self.get_metadata_value("save", False)
        verbose = self.get_metadata_value("verbose", False)
        data_yaml = self.get_metadata_value("data_yaml")

        self.model.train(  # type: ignore
            data=data_yaml,
            seed=seed,
            name=project_dir,
            val=False,
            verbose=verbose,
            save=save,
            **self.hparams,
        )

        return self

    def evaluate(self) -> Dict[str, float]:
        logger.info("Starting evaluation...")
        results = self.model.val(data=self.get_metadata_value("data_yaml"))  # type: ignore

        metrics = {
            "map_50": results.results_dict["metrics/mAP50(B)"],
            "map_50_95": results.results_dict["metrics/mAP50-95(B)"],
        }

        return metrics

    def predict(self, images: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError("Predict method is not yet implemented.")

    def save(self, dir: Path | str, prefix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model.pt"
        self.model.save(save_path)  # type: ignore
        return save_path


class YoloUltralyticsAdapter(UltralyticsAdapter):

    def setup(self) -> "YoloUltralyticsAdapter":
        model_version = self.get_metadata_value("weights")
        self.model = YOLO(model_version)
        self.model.to(self.device)

        return self

    def clone(self) -> "YoloUltralyticsAdapter":
        return YoloUltralyticsAdapter(
            classes=self.classes.copy(),
            metadata=self.metadata.copy(),
            hparams=self.hparams.copy(),
            device=self.device,
        )


class RtdetrUltralyticsAdapter(UltralyticsAdapter):

    def setup(self) -> "RtdetrUltralyticsAdapter":
        model_version = self.get_metadata_value("weights")
        self.model = RTDETR(model_version)
        self.model.to(self.device)

        return self

    def clone(self) -> "RtdetrUltralyticsAdapter":
        return RtdetrUltralyticsAdapter(
            classes=self.classes.copy(),
            metadata=self.metadata.copy(),
            hparams=self.hparams.copy(),
            device=self.device,
        )
