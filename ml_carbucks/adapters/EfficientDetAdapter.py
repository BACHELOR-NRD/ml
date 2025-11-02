from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.ops import nms
from effdet import create_model, create_loader
from effdet.data import resolve_input_config, resolve_fill_color
from effdet.bench import DetBenchPredict  # noqa F401
from effdet.anchors import Anchors, AnchorLabeler
from effdet.data.transforms import ResizePad, ImageToNumpy, Compose
from timm.optim._optim_factory import create_optimizer_v2


from ml_carbucks.utils.result_saver import ResultSaver
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

    def _predict_preprocess_images_v2(self, images: List[torch.Tensor]):
        input_config = resolve_input_config(self.get_params(), self.model.config)
        fill_color = resolve_fill_color(
            input_config["fill_color"], input_config["mean"]
        )
        transform = Compose(
            [
                ResizePad(
                    target_size=self.img_size,
                    interpolation=input_config["interpolation"],
                    fill_color=fill_color,
                ),
                ImageToNumpy(),
            ]
        )

        batch_list = []
        img_scaled = []

        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_proc, anno = transform(img_pil, dict())  # no annotations
            batch_list.append(img_proc)
            img_scaled.append(anno.get("img_scale", 1.0))

        batch_np = np.stack(batch_list, axis=0)  # [B,C,H,W]
        batch_tensor = torch.from_numpy(batch_np).float().to(self.device)

        return batch_tensor, img_scaled

    def predict(
        self,
        images: List[torch.Tensor],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
    ) -> List[ADAPTER_PREDICTION]:
        predictor = DetBenchPredict(deepcopy(self.model.model))
        predictor.to(self.device)
        predictor.eval()
        predictions: List[ADAPTER_PREDICTION] = []

        with torch.no_grad():
            batch_tensor, batch_scales = self._predict_preprocess_images_v2(images)
            outputs = predictor(batch_tensor)

            for i, pred in enumerate(outputs):
                out = pred.cpu()  # move to CPU
                boxes = out[:, :4]
                scores = out[:, 4]
                labels_idx = out[:, 5].long()

                # filter by confidence
                mask = scores >= conf_threshold
                boxes, scores, labels_idx = boxes[mask], scores[mask], labels_idx[mask]

                # apply NMS per image
                keep = nms(boxes, scores, iou_threshold)
                keep = keep[:max_detections]  # take top-k

                boxes = boxes[keep].numpy().copy()
                boxes *= batch_scales[i]

                scores = scores[keep].numpy()
                labels = [self.classes[idx - 1] for idx in labels_idx[keep]]

                predictions.append({"boxes": boxes, "scores": scores, "labels": labels})

        return predictions

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

        epochs = self.epochs
        opt = self.optimizer
        lr = self.lr
        weight_decay = self.weight_decay

        train_loader = self._create_loader(datasets, is_training=True)

        parser_max_label = train_loader.dataset.parsers[0].max_label  # type: ignore
        config_num_classes = self.model.config.num_classes

        if parser_max_label != config_num_classes:
            raise ValueError(
                f"Number of classes in dataset ({parser_max_label}) does not match "
                f"model config ({config_num_classes})."
                f"Please verify that the dataset is curated (classes IDs start from 1)"
            )

        optimizer = create_optimizer_v2(
            self.model,
            opt=opt,
            lr=lr,
            weight_decay=weight_decay,
        )

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")

            _ = self.train_epoch(optimizer, train_loader)  # type: ignore

        return self

    def _create_loader(
        self, datasets: List[Tuple[str | Path, str | Path]], is_training: bool
    ):
        batch_size = self.batch_size

        all_datasets = []
        for img_dir, ann_file in datasets:
            dataset = create_dataset_custom(
                img_dir=img_dir,
                ann_file=ann_file,
                has_labels=True,
            )
            all_datasets.append(dataset)

        concat_dataset = ConcatDetectionDataset(all_datasets)

        input_config = resolve_input_config(self.get_params(), self.model.config)
        loader = create_loader(
            concat_dataset,
            input_size=input_config["input_size"],
            batch_size=batch_size,
            is_training=is_training,
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

        return loader

    def train_epoch(
        self, optimizer: torch.optim.Optimizer, loader: DataLoader
    ) -> float:
        self.model.train()

        total_loss = 0.0
        for imgs, targets in tqdm(loader):
            output = self.model(imgs, targets)
            loss = output["loss"]
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss

    def debug(
        self,
        train_datasets: List[Tuple[str | Path, str | Path]],
        val_datasets: List[Tuple[str | Path, str | Path]],
        results_path: str | Path,
        results_name: str,
    ) -> Dict[str, float]:
        logger.info("Debugging training and evaluation loops...")

        epochs = self.epochs
        train_loader = self._create_loader(train_datasets, is_training=True)
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.optimizer,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        saver = ResultSaver(
            path=results_path,
            name=results_name,
        )
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")

            total_loss = self.train_epoch(optimizer, train_loader)  # type: ignore
            val_metrics = self.evaluate(val_datasets)
            saver.save(
                epoch=epoch,
                loss=total_loss,
                val_map=val_metrics["map_50_95"],
                val_map_50=val_metrics["map_50"],
            )
            logger.info(
                f"Debug Epoch {epoch}/{epochs} - Loss: {total_loss}, Val MAP: {val_metrics['map_50_95']}"
            )
            saver.plot(show=False)

        return val_metrics  # type: ignore

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> Dict[str, float]:
        self.model.eval()

        val_loader = self._create_loader(datasets, is_training=False)

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
