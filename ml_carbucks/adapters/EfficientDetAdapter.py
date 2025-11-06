from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from effdet import create_model, create_loader
from effdet.data import resolve_input_config, resolve_fill_color
from effdet.bench import DetBenchPredict  # noqa F401
from effdet.data.transforms import ResizePad, ImageToNumpy, Compose
from timm.optim._optim_factory import create_optimizer_v2

from ml_carbucks.utils.postprocessing import (
    postprocess_prediction_nms,
    postprocess_evaluation_results,
)
from ml_carbucks.utils.result_saver import ResultSaver
from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_METRICS,
    BaseDetectionAdapter,
    ADAPTER_PREDICTION,
)
from ml_carbucks.patches.effdet import (
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

    optimizer: str = "momentum"
    lr: float = 8e-3
    weight_decay: float = 9e-6
    confidence_threshold: float = 0.15
    training_augmentations: bool = True

    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model{suffix}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.model.state_dict(), save_path)
        return save_path

    def _preprocess_images(
        self, images: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_config = resolve_input_config(self.get_params(), self.model.config)
        fill_color = resolve_fill_color(
            input_config["fill_color"], input_config["mean"]
        )

        mean = (
            torch.tensor(input_config["mean"], device=self.device).view(3, 1, 1) * 255
        )
        std = torch.tensor(input_config["std"], device=self.device).view(3, 1, 1) * 255

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

        batch, scales = [], []

        for img in images:

            img_np, anno = transform(Image.fromarray(img), {})
            scales.append(anno.get("img_scale", 1.0))

            img_norm = (
                torch.from_numpy(img_np)
                .to(self.device, non_blocking=True)
                .float()
                .sub_(mean)
                .div_(std)
            )
            batch.append(img_norm)

        batch_tensor = torch.stack(batch, dim=0)  # B,C,H,W
        scales_tensor = torch.tensor(scales, dtype=torch.float32, device=self.device)
        batch_original_sizes = torch.tensor(
            [[img.shape[1], img.shape[0]] for img in images],
            dtype=torch.float32,
            device=self.device,
        )

        return batch_tensor, scales_tensor, batch_original_sizes

    def predict(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.15,
        iou_threshold: float = 1.0,
        max_detections: int = 10,
    ) -> List[ADAPTER_PREDICTION]:
        """
        The issue is that predicitons are weird but the results of the evaluation
        are good. So either the evaluation is wrong or the prediction extraction is wrong.
        """
        predictor = DetBenchPredict(deepcopy(self.model.model))
        predictor.to(self.device)
        predictor.eval()
        predictions: List[ADAPTER_PREDICTION] = []

        with torch.no_grad():

            batch_tensor, batch_scales, batch_original_sizes = self._preprocess_images(
                images
            )

            # NOTE: Passing the img_info dict will allow to get the predictions in original image scale
            img_info_dict = {
                "img_scale": batch_scales,
                "img_size": batch_original_sizes,
            }

            outputs = predictor(batch_tensor, img_info=img_info_dict)

            for i, pred in enumerate(outputs):
                boxes = pred[:, :4]
                scores = pred[:, 4]
                labels_idx = pred[:, 5]

                prediction = postprocess_prediction_nms(
                    boxes,
                    scores,
                    labels_idx,
                    conf_threshold,
                    iou_threshold,
                    max_detections,
                )

                predictions.append(prediction)

        return predictions

    def setup(self) -> "EfficientDetAdapter":
        img_size = self.img_size

        backbone = self.backbone
        weights = self.weights

        extra_args = dict(image_size=(img_size, img_size))
        self.model = create_model(
            model_name=backbone,
            bench_task="train",
            num_classes=len(self.classes),
            pretrained=weights == "",
            checkpoint_path=str(weights),
            # NOTE: we set it to True because we are using custom Mean Average Precision and it is easier that way
            # custom anchor labeler would be good idea if the boxes had unusual sizes and aspect ratios -> worth remembering for future
            bench_labeler=True,
            checkpoint_ema=False,
            **extra_args,
        )

        self.model.to(self.device)

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

        if is_training and (not self.training_augmentations):
            logger.warning(
                "Data augmentations are disabled. This may worsen model performance. It should only be used for debugging purposes."
            )

        input_config = resolve_input_config(self.get_params(), self.model.config)
        loader = create_loader(
            concat_dataset,
            input_size=input_config["input_size"],
            batch_size=batch_size,
            is_training=is_training and self.training_augmentations,
            use_prefetcher=True,
            interpolation=input_config["interpolation"],
            fill_color=input_config["fill_color"],
            mean=input_config["mean"],
            std=input_config["std"],
            num_workers=4,
            distributed=False,
            pin_mem=False,
            anchor_labeler=None,
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
        visualize: Literal["every", "last", "none"] = "none",
    ) -> ADAPTER_METRICS:
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
        val_metrics: Optional[ADAPTER_METRICS] = None
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

            show = False
            if visualize == "every":
                show = True
            elif visualize == "last" and epoch == epochs:
                show = True
            saver.plot(show=show)

        if val_metrics is None:
            raise RuntimeError("Validation metrics were not computed during debugging.")
        return val_metrics

    def evaluate(
        self,
        datasets: List[Tuple[str | Path, str | Path]],
        include_default: bool = False,
    ) -> ADAPTER_METRICS:
        self.model.eval()

        val_loader = self._create_loader(datasets, is_training=False)

        default_evaluator = None
        if include_default:
            default_evaluator = CocoStatsEvaluator(dataset=val_loader.dataset)

        evaluator = MeanAveragePrecision(extended_summary=False, class_metrics=False)
        total_loss = 0.0

        with torch.no_grad():
            for imgs, targets in val_loader:
                output = self.model(imgs, targets)
                loss = output["loss"]
                total_loss += loss.item()

                if include_default and default_evaluator is not None:
                    default_evaluator.add_predictions(
                        detections=output["detections"], target=targets
                    )

                # NOTE:
                # Annotations are loaded in yxyx format and they are scaled
                # Predicitons are in xyxy format and not scaled (original size of the image)
                # So we need to rescale the ground truth boxes to original sizes
                # Predicitons have a lot of low confidence scores and ground_truths have a lot of -1 values that just indicate no object
                # We need to filter them out
                for i in range(len(imgs)):
                    scale = targets["img_scale"][i]

                    pred_mask = (
                        output["detections"][i][:, 4] >= self.confidence_threshold
                    )
                    if pred_mask.sum() == 0:
                        # No predcitions above the confidence threshold
                        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                        pred_scores = torch.zeros((0,), dtype=torch.float32)
                        pred_labels = torch.zeros((0,), dtype=torch.int64)
                    else:
                        pred_boxes = output["detections"][i][pred_mask, :4]
                        pred_scores = output["detections"][i][pred_mask, 4]
                        pred_labels = output["detections"][i][pred_mask, 5].long()

                    gt_mask = targets["cls"][i] != -1
                    if gt_mask.sum() == 0:
                        # No ground truth boxes
                        gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                        gt_labels = torch.zeros((0,), dtype=torch.int64)
                    else:
                        gt_boxes_yxyx_raw = targets["bbox"][i][gt_mask]
                        gt_boxes_xyxy = torch.zeros_like(gt_boxes_yxyx_raw)
                        gt_boxes_xyxy[:, 0] = gt_boxes_yxyx_raw[:, 1]
                        gt_boxes_xyxy[:, 1] = gt_boxes_yxyx_raw[:, 0]
                        gt_boxes_xyxy[:, 2] = gt_boxes_yxyx_raw[:, 3]
                        gt_boxes_xyxy[:, 3] = gt_boxes_yxyx_raw[:, 2]
                        gt_boxes = gt_boxes_xyxy * scale
                        gt_labels = targets["cls"][i][gt_mask].long()

                    evaluator.update(
                        preds=[
                            {
                                "boxes": pred_boxes.cpu(),
                                "scores": pred_scores.cpu(),
                                "labels": pred_labels.cpu(),
                            }
                        ],
                        target=[
                            {
                                "boxes": gt_boxes.cpu(),
                                "labels": gt_labels.cpu(),
                            }
                        ],
                    )

        results = evaluator.compute()
        metrics = postprocess_evaluation_results(results)

        if include_default and default_evaluator is not None:
            default_results = default_evaluator.evaluate()
            metrics["default_map_50_95"] = default_results[0]  # type: ignore
            metrics["default_map_50"] = default_results[1]  # type: ignore

        return metrics
