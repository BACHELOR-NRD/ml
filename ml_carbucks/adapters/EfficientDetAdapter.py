from pathlib import Path
from typing import Any, Dict, List

import torch
from effdet import create_model, create_loader
from effdet.data import resolve_input_config
from effdet.anchors import Anchors, AnchorLabeler
from timm.optim._optim_factory import create_optimizer_v2
from tqdm import tqdm

from ml_carbucks.adapters.BaseDetectionAdapter import BaseDetectionAdapter
from ml_carbucks.utils.coco import CocoStatsEvaluator, create_dataset_custom
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


class EfficientDetAdapter(BaseDetectionAdapter):

    def load_model(self):
        logger.info("Loading EfficientDet model...")

        if self.model_path and self.model_path.exists():
            self._load_existing_model()
        else:
            self._create_model()

        self.model.to(self.device)

    def _load_existing_model(self):
        raise NotImplementedError(
            "Loading existing EfficientDet models is not yet implemented."
        )

    def _create_model(self):
        model_version = self.metadata.get("model_version", None)
        if not model_version:
            raise ValueError(
                "model_version must be specified in metadata to create a new model."
            )

        # Note: bench labeler can be both True and False, I know what difference it makes in a result
        # but I have no clue how does that difference works internally

        self.model = create_model(
            model_name=model_version,
            bench_task="train",
            num_classes=len(self.classes),
            pretrained=True,
            bench_labeler=False,
        )

    def _get_optimizer(self):

        if hasattr(self, "optimizer"):
            return self.optimizer

        opt = self.hparams["opt"]
        lr = self.hparams["lr"]
        weight_decay = self.hparams["weight_decay"]

        self.optimizer = create_optimizer_v2(
            self.model,
            opt=opt,
            lr=lr,
            weight_decay=weight_decay,
        )

        return self.optimizer

    def setup(self):

        train_img_dir = self.datasets.get("train_img_dir", None)
        train_ann_file = self.datasets.get("train_ann_file", None)

        val_img_dir = self.datasets.get("val_img_dir", None)
        val_ann_file = self.datasets.get("val_ann_file", None)

        if (
            not train_img_dir
            or not train_ann_file
            or not val_img_dir
            or not val_ann_file
        ):
            raise ValueError(
                "Both train and val dataset paths must be provided in datasets to load data."
            )

        input_config = resolve_input_config(dict(), self.model.config)

        dataset_train = create_dataset_custom(
            img_dir=train_img_dir,
            ann_file=train_ann_file,
            has_labels=True,
        )

        dataset_val = create_dataset_custom(
            img_dir=val_img_dir,
            ann_file=val_ann_file,
            has_labels=True,
        )

        labeler = AnchorLabeler(
            Anchors.from_config(self.model.config),
            self.model.config.num_classes,
            match_threshold=0.5,
        )

        batch_size = self.hparams["batch_size"]

        self.train_loader = create_loader(
            dataset_train,
            input_size=input_config["input_size"],
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            # NOTE: currrently not used
            # re_prob=args.reprob,
            # re_mode=args.remode,
            # re_count=args.recount,
            interpolation=input_config["interpolation"],
            fill_color=input_config["fill_color"],
            mean=input_config["mean"],
            std=input_config["std"],
            num_workers=4,
            distributed=False,
            pin_mem=False,
            anchor_labeler=labeler,
            transform_fn=None,
            collate_fn=None,
        )

        self.val_loader = create_loader(
            dataset_val,
            input_size=input_config["input_size"],
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation=input_config["interpolation"],
            fill_color=input_config["fill_color"],
            mean=input_config["mean"],
            std=input_config["std"],
            num_workers=4,
            distributed=False,
            pin_mem=False,
            anchor_labeler=labeler,
            transform_fn=None,
            collate_fn=None,
        )

        self.evaluator = CocoStatsEvaluator(self.val_loader.dataset)

        parser_max_label = self.train_loader.dataset.parser.max_label  # type: ignore
        config_num_classes = self.model.config.num_classes

        if parser_max_label != config_num_classes:
            raise ValueError(
                f"Number of classes in dataset ({parser_max_label}) does not match "
                f"model config ({config_num_classes})."
                f"Please verify that the dataset is curated (classes IDs start from 1)"
            )

    def fit(self):
        logger.info("Starting training...")

        epochs = self.hparams["epochs"]

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.train_epoch()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        optimizer = self._get_optimizer()

        for imgs, targets in tqdm(self.train_loader):
            output = self.model(imgs, targets)
            loss = output["loss"]
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        self.evaluator.reset()

        total_loss = 0.0
        with torch.no_grad():
            for imgs, targets in self.val_loader:

                output = self.model(imgs, targets)
                loss = output["loss"]
                total_loss += loss.item()
                self.evaluator.add_predictions(output["detections"], targets)

        results = self.evaluator.evaluate()

        metrics = {
            "map_50": results[1],
            "map_50_95": results[0],
        }

        return metrics

    def predict(self, images: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError("Predict method is not yet implemented.")

    def save_model(self, dir: Path | str) -> Path:
        save_path = Path(dir) / "model.pth"
        torch.save(self.model.state_dict(), save_path)
        return save_path
