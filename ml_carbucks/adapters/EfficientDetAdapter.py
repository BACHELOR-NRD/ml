from pathlib import Path
from typing import Any, Dict, List

import torch
from effdet import create_model, create_loader
from effdet.data import resolve_input_config
from effdet.anchors import Anchors, AnchorLabeler
from timm.optim._optim_factory import create_optimizer_v2
from tqdm import tqdm

from ml_carbucks.adapters.BaseDetectionAdapter import (
    BaseDetectionAdapter,
    ADAPTER_PREDICTION,
)
from ml_carbucks.utils.coco import CocoStatsEvaluator, create_dataset_custom
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


class EfficientDetAdapter(BaseDetectionAdapter):

    def get_possible_hyper_keys(self) -> List[str]:
        return [
            "img_size",
            "batch_size",
            "epochs",
            "opt",
            "lr",
            "weight_decay",
        ]

    def get_required_metadata_keys(self) -> List[str]:
        return [
            "version",
            "train_img_dir",
            "train_ann_file",
            "val_img_dir",
            "val_ann_file",
        ]

    def save(self, dir: Path | str, prefix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model.pth"
        torch.save(self.model.model.state_dict(), save_path)
        return save_path

    def clone(self) -> "EfficientDetAdapter":
        return EfficientDetAdapter(
            classes=self.classes,
            metadata=self.metadata.copy(),
            hparams=self.hparams.copy(),
            device=self.device,
        )

    def predict(self, images: Any) -> List[ADAPTER_PREDICTION]:
        raise NotImplementedError("Predict method is not yet implemented.")

    def setup(self) -> "EfficientDetAdapter":
        img_size = self.get_param("img_size")

        version = self.get_metadata_value("version")
        weights = self.get_metadata_value("weights", None)
        bench_labeler = self.get_metadata_value("bench_labeler", True)

        # NOTE: img size would need to be updated here if we want to change it
        # I dont think it is possible to change it after model creation
        extra_args = dict(image_size=(img_size, img_size))
        self.model = create_model(
            model_name=version,
            bench_task="train",
            num_classes=len(self.classes),
            pretrained=weights is None,
            checkpoint_path=weights,
            bench_labeler=bench_labeler,
            checkpoint_ema=False,
            **extra_args,
        )

        self.model.to(self.device)

        self.labeler = None
        if bench_labeler is False:
            self.labeler = AnchorLabeler(
                Anchors.from_config(self.model.config),
                self.model.config.num_classes,
                match_threshold=0.5,
            )

        return self

    def fit(self) -> "EfficientDetAdapter":
        logger.info("Starting training...")
        self.model.train()

        batch_size = self.get_param("batch_size")
        epochs = self.get_param("epochs")
        opt = self.get_param("opt", "momentum")
        lr = self.get_param("lr", 7e-3)
        weight_decay = self.get_param("weight_decay", 1e-5)

        train_img_dir = self.get_metadata_value("train_img_dir")
        train_ann_file = self.get_metadata_value("train_ann_file")

        input_config = resolve_input_config(self.hparams, self.model.config)

        train_dataset = create_dataset_custom(
            img_dir=train_img_dir,
            ann_file=train_ann_file,
            has_labels=True,
        )

        train_loader = create_loader(
            train_dataset,
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
            anchor_labeler=self.labeler,
            transform_fn=None,
            collate_fn=None,
        )

        parser_max_label = train_loader.dataset.parser.max_label  # type: ignore
        config_num_classes = self.model.config.num_classes

        if parser_max_label != config_num_classes:
            raise ValueError(
                f"Number of classes in dataset ({parser_max_label}) does not match "
                f"model config ({config_num_classes})."
                f"Please verify that the dataset is curated (classes IDs start from 1)"
            )

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            total_loss = 0.0

            optimizer = create_optimizer_v2(
                self.model,
                opt=opt,
                lr=lr,
                weight_decay=weight_decay,
            )

            for imgs, targets in tqdm(train_loader):
                output = self.model(imgs, targets)
                loss = output["loss"]
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        batch_size = self.get_param("batch_size")

        val_img_dir = self.get_metadata_value("val_img_dir")
        val_ann_file = self.get_metadata_value("val_ann_file")

        dataset_val = create_dataset_custom(
            img_dir=val_img_dir,
            ann_file=val_ann_file,
            has_labels=True,
        )

        input_config = resolve_input_config(self.hparams, self.model.config)

        val_loader = create_loader(
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
            anchor_labeler=self.labeler,
            transform_fn=None,
            collate_fn=None,
        )

        evaluator = CocoStatsEvaluator(val_loader.dataset)
        total_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                output = self.model(imgs, targets)
                loss = output["loss"]
                total_loss += loss.item()
                evaluator.add_predictions(output["detections"], targets)

        results = evaluator.evaluate()
        metrics = {
            "map_50": results[1],
            "map_50_95": results[0],
        }
        return metrics
