from copy import deepcopy
import torch
from torch import Tensor
from torchvision.ops import nms

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_PREDICTION,
    ADAPTER_METRICS,
)


def convert_pred2eval(pred: ADAPTER_PREDICTION) -> ADAPTER_PREDICTION:
    """Detach adapter outputs so downstream metrics stay on CPU."""
    return {
        "boxes": pred["boxes"].clone().detach().cpu(),
        "scores": pred["scores"].clone().detach().cpu(),
        "labels": pred["labels"].clone().detach().cpu().long(),
    }


def postprocess_prediction_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> ADAPTER_PREDICTION:
    """A function that applies Non-Maximum Suppression (NMS) to model predictions.


    Args:
        boxes (Tensor): A tensor of shape (N, 4) representing bounding box coordinates.
        scores (Tensor): A tensor of shape (N,) representing confidence scores for each bounding box.
        labels (Tensor): A tensor of shape (N,) representing class labels for each bounding box.
        conf_threshold (float): Confidence threshold to filter out low-confidence boxes.
        iou_threshold (float): Intersection-over-Union (IoU) threshold for NMS.
        max_detections (int): Maximum number of detections to keep after NMS.

    Raises:
        ValueError: If max_detections is not an integer or None.

    Returns:
        ADAPTER_PREDICTION: A dictionary containing the post-processed predictions with keys "boxes", "scores", and "labels".
            It is detached, cloned, and on CPU.
    """

    device = boxes.device
    boxes = boxes.to(device)
    scores = scores.to(device)
    labels = labels.to(device)

    if boxes.numel() == 0 or scores.numel() == 0 or labels.numel() == 0:
        return {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.long),
        }

    if conf_threshold is not None and conf_threshold > 0:
        keep_mask = scores >= float(conf_threshold)
        if keep_mask.sum().item() == 0:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

    if max_detections is None:
        max_detections = 0
    if not isinstance(max_detections, int):
        try:
            max_detections = int(max_detections)
        except Exception:
            raise ValueError("max_detections must be an integer or None")

    # If no NMS requested (iou out of (0,1) range), just select top-K by score
    if not (0.0 < float(iou_threshold) < 1.0):
        if max_detections <= 0:
            # return all (already on CPU)
            order = scores.argsort(descending=True)
        else:
            order = scores.argsort(descending=True)[:max_detections]
        return {
            "boxes": boxes[order].clone().detach(),
            "scores": scores[order].clone().detach(),
            "labels": labels[order].clone().detach().long(),
        }

    # Per-class NMS
    keep_indices_list = []
    unique_labels = torch.unique(labels)
    for cls in unique_labels:
        cls_mask = labels == cls
        if cls_mask.sum() == 0:
            continue
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        # nms expects tensors on same device (we're on CPU), and returns indices relative to cls_boxes
        cls_keep = nms(cls_boxes, cls_scores, float(iou_threshold))
        if cls_keep.numel() == 0:
            continue
        # Map class-local indices back to global indices
        global_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
        keep_indices_list.append(global_indices[cls_keep])

    if len(keep_indices_list) == 0:
        return {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.long),
        }

    keep_indices = torch.cat(keep_indices_list)

    # Sort kept boxes by descending score and enforce max_detections
    sorted_by_score = scores[keep_indices].argsort(descending=True)
    if max_detections > 0:
        sorted_by_score = sorted_by_score[:max_detections]
    final_indices = keep_indices[sorted_by_score]

    final_boxes = boxes[final_indices].clone().detach()
    final_scores = scores[final_indices].clone().detach()
    final_labels = labels[final_indices].clone().detach().long()

    return {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}


def postprocess_evaluation_results(metrics: dict[str, Tensor]) -> ADAPTER_METRICS:
    """
    Process evaluation results from Mean Average Precision metric.
    This was created to unify the output format from

    Args:
        metrics (dict): A dictionary containing evaluation metrics.

    Returns:
        dict: A dictionary with processed 'map_50' and 'map' as map_50_95,
    """
    return {
        "map_50": metrics["map_50"].item(),
        "map_50_95": metrics["map"].item(),
        "map_75": metrics["map_75"].item(),
        "classes": metrics["classes"].tolist() if "classes" in metrics else [],
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
        ADAPTER_PREDICTION: fused boxes, scores, and labels. It is detached, cloned, and on CPU.
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
        "boxes": fused_boxes.detach().cpu(),
        "scores": fused_scores.detach().cpu(),
        "labels": fused_labels.detach().cpu().long(),
    }


def map_predictions_labels(
    predictions: list[ADAPTER_PREDICTION], label_mapper: dict[int, int]
) -> list[ADAPTER_PREDICTION]:
    """
    Map prediction labels using a provided label mapping dictionary.

    Args:
        predictions (ADAPTER_PREDICTION): The original predictions with labels to be mapped.
        label_mapper (dict[int, int]): A dictionary mapping original labels to new labels.

    Returns:
        ADAPTER_PREDICTION: Predictions with mapped labels.
    """

    mapped_predictions = deepcopy(predictions)
    for pred in mapped_predictions:
        mapped_labels = []
        for label in pred["labels"]:
            mapped_label = label_mapper.get(int(label.item()), int(label.item()))
            mapped_labels.append(mapped_label)
        pred["labels"] = torch.tensor(mapped_labels, dtype=torch.long)

    return mapped_predictions
