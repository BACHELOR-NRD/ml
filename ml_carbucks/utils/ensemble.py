from copy import deepcopy
from typing import Literal, Optional

import torch

from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import postprocess_prediction
from ml_carbucks.adapters.BaseDetectionAdapter import ADAPTER_PREDICTION

logger = setup_logger(__name__)


def normalize_scores(
    preds_list: list[list[torch.Tensor]],
    method: Literal["minmax", "zscore"] = "minmax",
    trust: Optional[list[float]] = None,
) -> list[torch.Tensor]:
    """
    Normalize confidence scores of predictions from multiple adapters.
    Args:
        preds_list (list): List of predictions from different adapters. Each prediction is a list of tensors per image.
        method (str): Normalization method, either "minmax" or "zscore".
        trust (list, optional): Trust weights for each adapter. If None, equal weights are used.
    Returns:
        list: Normalized predictions with adjusted confidence scores.
    """

    if trust is None:
        trust = [1.0] * len(preds_list)
    normalized_all = []
    for preds, t in zip(preds_list, trust):
        flat_scores = torch.cat([p[:, 4] for p in preds], dim=0)
        s_min, s_max = flat_scores.min(), flat_scores.max()
        mean, std = flat_scores.mean(), flat_scores.std() + 1e-6
        normalized = deepcopy(preds)

        for p in normalized:
            if method == "minmax":
                p[:, 4] = (p[:, 4] - s_min) / (s_max - s_min + 1e-6)
            elif method == "zscore":
                p[:, 4] = (p[:, 4] - mean) / std
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            p[:, 4] = p[:, 4] * t

        normalized_all.append(normalized)
    return normalized_all


def weighted_boxes_fusion(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.55,
    conf_threshold: float = 0.001,
    max_detections: int = 300,
) -> ADAPTER_PREDICTION:
    """
    Perform Weighted Boxes Fusion (WBF) on a single image's predictions.

    Args:
        boxes (Tensor): [N, 4] boxes in [x1, y1, x2, y2].
        scores (Tensor): [N] confidence scores.
        labels (Tensor): [N] class indices.
        iou_threshold (float): IoU threshold for merging boxes.
        conf_threshold (float): Minimum score to keep.
        max_detections (int): Max number of output boxes.

    Returns:
        ADAPTER_PREDICTION: fused boxes, scores, and labels.
    """

    # Filter by confidence
    keep = scores >= conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if boxes.numel() == 0:
        return {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.long),
        }

    # Convert to float32 for safety
    boxes = boxes.float()
    scores = scores.float()

    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for cls in labels.unique():
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        if cls_boxes.size(0) == 0:
            continue

        # Sort by confidence descending
        order = torch.argsort(cls_scores, descending=True)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        fused_cls_boxes = []
        fused_cls_scores = []

        while len(cls_boxes) > 0:
            # Pick the top-scoring box
            main_box = cls_boxes[0]
            main_score = cls_scores[0]

            if len(cls_boxes) == 1:
                fused_cls_boxes.append(main_box)
                fused_cls_scores.append(main_score)
                break

            # Compute IoU between top box and the rest
            ious = box_iou(main_box.unsqueeze(0), cls_boxes[1:]).squeeze(0)
            overlap_mask = ious > iou_threshold

            overlapping_boxes = cls_boxes[1:][overlap_mask]
            overlapping_scores = cls_scores[1:][overlap_mask]

            # Include the main box itself
            all_boxes = torch.cat([main_box.unsqueeze(0), overlapping_boxes], dim=0)
            all_scores = torch.cat([main_score.unsqueeze(0), overlapping_scores], dim=0)

            # Weighted average of coordinates
            weights = all_scores / all_scores.sum()
            fused_box = (all_boxes * weights[:, None]).sum(dim=0)

            fused_score = all_scores.mean()  # or sum() / len()
            fused_cls_boxes.append(fused_box)
            fused_cls_scores.append(fused_score)

            # Remove used boxes
            keep_mask = torch.ones(len(cls_boxes), dtype=torch.bool)
            keep_mask[1:][overlap_mask] = False
            keep_mask[0] = False
            cls_boxes = cls_boxes[keep_mask]
            cls_scores = cls_scores[keep_mask]

        fused_boxes.extend(fused_cls_boxes)
        fused_scores.extend(fused_cls_scores)
        fused_labels.extend([cls] * len(fused_cls_boxes))

    if len(fused_boxes) == 0:
        return {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.long),
        }

    # Convert lists to tensors
    fused_boxes = torch.stack(fused_boxes)
    fused_scores = torch.tensor(fused_scores)
    fused_labels = torch.tensor(
        [int(cls.item()) for cls in fused_labels], dtype=torch.long
    )

    # Sort final results
    order = torch.argsort(fused_scores, descending=True)
    fused_boxes = fused_boxes[order][:max_detections]
    fused_scores = fused_scores[order][:max_detections]
    fused_labels = fused_labels[order][:max_detections]

    return {
        "boxes": fused_boxes,
        "scores": fused_scores,
        "labels": fused_labels,
    }


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes."""
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def fuse_adapters_predictions(
    adapters_predictions: list[list[ADAPTER_PREDICTION]],
    max_detections: int,
    iou_threshold: float,
    conf_threshold: float,
    strategy: Optional[Literal["nms", "wbf"]] = None,
) -> list[ADAPTER_PREDICTION]:
    """
    Fuse per-image predictions from multiple adapters into a single list of ADAPTER_PREDICTIONs.
    """

    num_images = len(adapters_predictions[0])
    list_of_tensors_per_adapter_org = [
        [
            (
                torch.cat(
                    [
                        p["boxes"],
                        p["scores"].unsqueeze(1),
                        p["labels"].unsqueeze(1).float(),
                    ],
                    dim=1,
                )
                if len(p["boxes"]) > 0
                else torch.empty((0, 6))
            )
            for p in preds_per_adapter
        ]
        for preds_per_adapter in adapters_predictions
    ]

    combined_list_of_tensors = []
    for img_idx in range(num_images):
        combined = torch.cat(
            [
                list_of_tensors_per_adapter_org[adapter_i][img_idx]
                for adapter_i in range(len(adapters_predictions))
            ],
            dim=0,
        )

        # sort by confidence score (column 4) in descending order
        if combined.numel() > 0:
            sorted_idx = combined[:, 4].argsort(descending=True)
            combined = combined[sorted_idx]

        combined_list_of_tensors.append(combined)

    strategy_predictions = []
    if strategy == "nms":
        logger.info("Applying NMS fusion strategy...")
        for combined_preds in combined_list_of_tensors:
            if combined_preds.numel() == 0:
                nms_combined: ADAPTER_PREDICTION = {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "labels": torch.empty((0,), dtype=torch.long),
                }
            else:
                pred_boxes = combined_preds[:, :4]
                pred_scores = combined_preds[:, 4]
                pred_labels = combined_preds[:, 5]
                nms_combined = postprocess_prediction(
                    boxes=pred_boxes,
                    scores=pred_scores,
                    labels=pred_labels,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold,
                    max_detections=max_detections,
                )
            strategy_predictions.append(nms_combined)
    elif strategy == "wbf":
        logger.info("Applying WBF fusion strategy...")
        for combined_preds in combined_list_of_tensors:
            if combined_preds.numel() == 0:
                wbf_combined: ADAPTER_PREDICTION = {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "labels": torch.empty((0,), dtype=torch.long),
                }
            else:
                pred_boxes = combined_preds[:, :4]
                pred_scores = combined_preds[:, 4]
                pred_labels = combined_preds[:, 5]
                wbf_combined = weighted_boxes_fusion(
                    boxes=pred_boxes,
                    scores=pred_scores,
                    labels=pred_labels,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold,
                    max_detections=max_detections,
                )
            strategy_predictions.append(wbf_combined)
    else:
        strategy_list_of_tensors = [
            combined_preds[combined_preds[:, 4] >= conf_threshold]
            .clone()
            .detach()[:max_detections]
            for combined_preds in combined_list_of_tensors
        ]
        for tpreds in strategy_list_of_tensors:
            prediction: ADAPTER_PREDICTION = {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }

            if tpreds.numel() > 0:
                prediction = {
                    "boxes": tpreds[:, :4],
                    "scores": tpreds[:, 4],
                    "labels": tpreds[:, 5].long(),
                }

            strategy_predictions.append(prediction)

        logger.warning(f"Unknown fusion strategy: {strategy}, skipping fusion.")
    return strategy_predictions
