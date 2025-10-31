from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from effdet.data import resolve_input_config
from effdet import create_model, create_loader
from effdet.anchors import Anchors, AnchorLabeler
from timm.optim._optim_factory import create_optimizer_v2

from ml_carbucks.adapters.BaseDetectionAdapter import (
    BaseDetectionAdapter,
    ADAPTER_PREDICTION,
)
from ml_carbucks.utils.effdet_extension import (
    CocoStatsEvaluator,
    ConcatDetectionDataset,
    create_dataset_custom,
)
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EfficientDetAdapter(BaseDetectionAdapter):

    weights: str | Path = ""
    backbone: str = "tf_efficientdet_d0"
    bench_labeler: bool = False

    optimizer: str = "momentum"
    lr: float = 8e-3
    weight_decay: float = 5e-5

    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model{suffix}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.model.state_dict(), save_path)
        return save_path

    def clone(self) -> "EfficientDetAdapter":
        return EfficientDetAdapter(
            classes=deepcopy(self.classes),
            weights=self.weights,
            img_size=self.img_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            backbone=self.backbone,
            bench_labeler=self.bench_labeler,
            optimizer=self.optimizer,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def predict(self, images: List[torch.Tensor]) -> List[ADAPTER_PREDICTION]:
        raise NotImplementedError("Predict method is not yet implemented.")

    def setup(self) -> "EfficientDetAdapter":
        img_size = self.img_size

        backbone = self.backbone
        weights = self.weights
        bench_labeler = self.bench_labeler

        # NOTE: img size would need to be updated here if we want to change it
        # I dont think it is possible to change it after model creation
        extra_args = dict(image_size=(img_size, img_size))
        self.model = create_model(
            model_name=backbone,
            bench_task="train",
            num_classes=len(self.classes),
            pretrained=weights == "",
            checkpoint_path=str(weights),
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

    def fit(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> "EfficientDetAdapter":
        logger.info("Starting training...")
        self.model.train()

        batch_size = self.batch_size
        epochs = self.epochs
        opt = self.optimizer
        lr = self.lr
        weight_decay = self.weight_decay

        all_datasets = []
        for img_dir, ann_file in datasets:
            dataset = create_dataset_custom(
                img_dir=img_dir,
                ann_file=ann_file,
                has_labels=True,
            )
            all_datasets.append(dataset)

        train_dataset = ConcatDetectionDataset(all_datasets)

        input_config = resolve_input_config(dict(), self.model.config)
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

        parser_max_label = train_loader.dataset.parsers[0].max_label  # type: ignore
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

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> Dict[str, float]:
        self.model.eval()

        batch_size = self.batch_size

        all_datasets = []
        for img_dir, ann_file in datasets:
            dataset = create_dataset_custom(
                img_dir=img_dir,
                ann_file=ann_file,
                has_labels=True,
            )
            all_datasets.append(dataset)

        dataset_val = ConcatDetectionDataset(all_datasets)

        input_config = resolve_input_config(dict(), self.model.config)
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
