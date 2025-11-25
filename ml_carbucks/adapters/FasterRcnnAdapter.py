from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_METRICS,
    ADAPTER_CHECKPOINT,
    ADAPTER_PREDICTION,
    BaseDetectionAdapter,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import (
    convert_pred2eval,
    postprocess_prediction_nms,
    weighted_boxes_fusion,
    postprocess_evaluation_results,
)
from ml_carbucks.utils.preprocessing import (
    create_clean_loader,
    create_transforms,
    preprocess_images,
)
from ml_carbucks.utils.result_saver import ResultSaver

logger = setup_logger(__name__)

FASTERRCNN_OPTIMIZER_OPTIONS = Literal["Adam", "AdamW", "RAdam", "SGD"]


@dataclass
class FasterRcnnAdapter(BaseDetectionAdapter):

    # --- HYPER PARAMETERS ---

    # lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    # weight_decay_backbone: float = 1e-5
    weight_decay_head: float = 1e-4
    optimizer: FASTERRCNN_OPTIMIZER_OPTIONS = "AdamW"
    clip_gradients: Optional[float] = None
    momentum: float = 0.9  # Used for SGD and RMSprop
    strategy: Literal["nms", "wbf"] = "nms"
    accumulation_steps: int = 1

    # --- SETUP PARAMETERS ---

    n_classes: int = 3
    training_augmentations: bool = True

    # --- MAIN METHODS ---

    def _setup(self) -> "FasterRcnnAdapter":
        if self.weights == "DEFAULT":
            self.weights = "V1"

        if self.checkpoint is not None:
            self._load_from_checkpoint(self.checkpoint)  # type: ignore

        elif self.weights in ("V1", "V2"):

            self._create_model_wrapper(
                weights=str(self.weights),
                img_size=self.img_size,
                n_classes=self.n_classes,
            )

        else:
            raise ValueError(
                "Weights must be 'DEFAULT', 'V1', 'V2' or a pickled checkpoint"
            )

        self.model.to(self.device)

        return self

    def fit(self, datasets: List[Tuple[str | Path, str | Path]]) -> "FasterRcnnAdapter":
        logger.info("Starting training...")

        epochs = self.epochs

        loader = self._create_loader(datasets, is_training=True)

        optimizer = self._create_optimizer()

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")

            _ = self.train_epoch(optimizer, loader)

        return self

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        optimizer.zero_grad()
        cnt = 0
        for imgs, targets in tqdm(
            loader, desc="Training", unit="batch", disable=not self.verbose
        ):
            imgs = list(img.to(self.device) for img in imgs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()  # type: ignore

            (loss / self.accumulation_steps).backward()  # type: ignore

            if (cnt + 1) % self.accumulation_steps == 0:
                self._clip_gradients_wrapper()
                optimizer.step()
                optimizer.zero_grad()

            cnt += 1

        if (cnt) % self.accumulation_steps != 0:
            self._clip_gradients_wrapper()
            optimizer.step()
            optimizer.zero_grad()

        return total_loss

    def evaluate(
        self, datasets: List[Tuple[str | Path, str | Path]]
    ) -> ADAPTER_METRICS:
        logger.info("Starting evaluation...")
        self.model.eval()

        loader = self._create_loader(datasets, is_training=False)

        evaluator = MeanAveragePrecision(extended_summary=False, class_metrics=False)
        with torch.no_grad():
            for imgs, targets in tqdm(
                loader, desc="Evaluating", unit="batch", disable=not self.verbose
            ):
                imgs = list(img.to(self.device) for img in imgs)
                outputs = self.model(imgs)

                # Move targets and outputs to CPU for metric computation
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                outputs_cpu = [{k: v.cpu() for k, v in o.items()} for o in outputs]

                predictions_processed = []
                for output in outputs_cpu:
                    boxes = output["boxes"]
                    scores = output["scores"]
                    labels = output["labels"]

                    # NOTE: those threshold values are hardcoded for evaluation but could be parameterized if needed
                    if self.strategy == "nms":
                        processed_pred = postprocess_prediction_nms(
                            boxes,
                            scores,
                            labels,
                            conf_threshold=0.02,
                            iou_threshold=0.7,
                            max_detections=300,
                        )
                    elif self.strategy == "wbf":
                        processed_pred = weighted_boxes_fusion(
                            boxes,
                            scores,
                            labels,
                            conf_threshold=0.02,
                            iou_threshold=0.7,
                            max_detections=300,
                        )
                    else:
                        raise ValueError(f"Unsupported strategy: {self.strategy}")

                    predictions_processed.append(convert_pred2eval(processed_pred))

                evaluator.update(predictions_processed, targets_cpu)

        results = evaluator.compute()

        metrics = postprocess_evaluation_results(results)

        return metrics

    def predict(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_detections: int = 10,
    ) -> List[ADAPTER_PREDICTION]:
        self.model.eval()

        preprocessed_images, scales, original_sizes = preprocess_images(
            images, img_size=self.img_size
        )

        images_fasterrcnn = [img.to(self.device) for img in preprocessed_images]

        with torch.no_grad():
            outputs = self.model(images_fasterrcnn)

        processed_predictions: List[ADAPTER_PREDICTION] = []
        for i, output in enumerate(outputs):
            boxes = output["boxes"] / scales[i]
            scores = output["scores"]
            labels = output["labels"]

            if self.strategy == "nms":
                prediction = postprocess_prediction_nms(
                    boxes,
                    scores,
                    labels,
                    conf_threshold,
                    iou_threshold,
                    max_detections,
                )
            elif self.strategy == "wbf":
                prediction = weighted_boxes_fusion(
                    boxes,
                    scores,
                    labels,
                    conf_threshold,
                    iou_threshold,
                    max_detections,
                )
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

            processed_predictions.append(prediction)

        return processed_predictions

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
        optimizer = self._create_optimizer()
        saver = ResultSaver(results_path, name=results_name)

        val_metrics: Optional[ADAPTER_METRICS] = None
        for epoch in range(1, epochs + 1):
            logger.info(f"Debug Epoch {epoch}/{epochs}")
            total_loss = self.train_epoch(optimizer, train_loader)
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
            saver.plot(show=show, secondaries_y=["val_map", "val_map_50"])

        if val_metrics is None:
            raise RuntimeError("Validation metrics were not computed during debugging.")
        return val_metrics

    def save(self, dir: Path | str, prefix: str = "", suffix: str = "") -> Path:
        save_path = Path(dir) / f"{prefix}model{suffix}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        params = self.get_params(skip=["checkpoint"])

        obj = {
            "class_data": {
                "name": self.__class__.__name__,
                "module": self.__class__.__module__,
                "class_type": self.__class__,
            },
            "params": params,
            "model": self.model.state_dict(),
        }

        pkl.dump(obj, open(save_path, "wb"))

        return save_path

    # --- HELPER METHODS ---

    def _load_from_checkpoint(
        self, checkpoint_path: str | Path | dict, **kwargs
    ) -> None:
        if isinstance(checkpoint_path, dict):
            obj: ADAPTER_CHECKPOINT = checkpoint_path  # type: ignore
        else:
            obj: ADAPTER_CHECKPOINT = pkl.load(open(checkpoint_path, "rb"))

        obj_class_name = obj["class_data"]["name"]
        if obj_class_name != self.__class__.__name__:
            raise ValueError(
                f"Pickled adapter class mismatch: expected '{self.__class__.__name__}', got '{obj_class_name}'"
            )

        params = obj["params"]

        logger.warning("Overwriting adapter parameters with loaded pickled parameters.")

        self.set_params(params)

        self._create_model_wrapper(
            weights=str(self.weights),
            img_size=self.img_size,
            n_classes=self.n_classes,
        )

        self.model.load_state_dict(obj["model"])

    def _clip_gradients_wrapper(self) -> None:
        if self.clip_gradients is not None:
            clip_grad_norm_(self.model.parameters(), max_norm=self.clip_gradients)

    def _create_model_wrapper(
        self, weights: str, img_size: int, n_classes: int
    ) -> None:

        if weights == "V1":
            weights_enum = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(
                weights=weights_enum,
                min_size=img_size,
                max_size=img_size,
            )

        elif weights == "V2":
            weights_enum = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights=weights_enum,
                min_size=img_size,
                max_size=img_size,
            )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, n_classes + 1  # +1 for background
        )

    def _create_optimizer(self):
        # NOTE: Using different learning rates and weight decays for backbone and head
        # common practice to hard-code lower lr and weight decay for backbone
        # could be parameterized if needed

        # lr1 = self.lr_backbone
        lr1 = self.lr_head / 10
        lr2 = self.lr_head
        # weight_decay1 = self.weight_decay_backbone
        weight_decay1 = self.weight_decay_head / 10
        weight_decay2 = self.weight_decay_head

        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            elif "roi_heads" in name:
                head_params.append(param)

        params = [
            {"params": backbone_params, "lr": lr1, "weight_decay": weight_decay1},
            {"params": head_params, "lr": lr2, "weight_decay": weight_decay2},
        ]

        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params)
        elif self.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(params)
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(params)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(params, momentum=self.momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        return optimizer

    def _create_loader(
        self, datasets: List[Tuple[str | Path, str | Path]], is_training: bool
    ) -> DataLoader:
        batch_size = self.batch_size
        img_size = self.img_size

        if is_training and (not self.training_augmentations):
            logger.warning(
                "Data augmentations are disabled. This may worsen model performance. It should only be used for debugging purposes."
            )

        return create_clean_loader(
            datasets,
            shuffle=is_training and self.training_augmentations,
            batch_size=batch_size,
            transforms=create_transforms(
                is_training=is_training and self.training_augmentations,
                img_size=img_size,
            ),
        )
