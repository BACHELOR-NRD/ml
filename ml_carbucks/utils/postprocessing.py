import torch
from torch import Tensor
from torchvision.ops import nms

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_PREDICTION,
    ADAPTER_METRICS,
)


def postprocess_prediction_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> ADAPTER_PREDICTION:

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
