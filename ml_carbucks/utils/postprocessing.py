from typing import Any
import torch
from torch import Tensor
from torchvision.ops import nms

from ml_carbucks.adapters.BaseDetectionAdapter import ADAPTER_PREDICTION


def postprocess_prediction(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> ADAPTER_PREDICTION:
    """
    A function that postprocesses model predictions by:
     - applying confidence thresholding
     - applying non-maximum suppression (NMS)
     - limiting the number of detections to max_detections

    If confidence threshold <= 0, no thresholding is applied.
    If IoU threshold is out of range (0, 1), NMS is skipped.
    If max_detections is less than or equal to 0, all detections are kept after thresholding/NMS.
    """

    # Apply confidence threshold
    mask = scores >= conf_threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    if boxes.numel() == 0:
        return {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,)).long(),
        }

    if max_detections <= 0:
        max_detections = boxes.shape[0]
    else:
        max_detections = min(max_detections, boxes.shape[0])

    if iou_threshold <= 0 or iou_threshold >= 1.0:
        # No NMS, just limit to max_detections

        topk_indices = scores.topk(max_detections).indices
        top_boxes = boxes[topk_indices]
        top_scores = scores[topk_indices]
        top_labels = labels[topk_indices]

        return {
            "boxes": top_boxes.cpu(),
            "scores": top_scores.cpu(),
            "labels": top_labels.cpu().long(),
        }

    # Apply NMS per class
    keep_indices = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = nms(cls_boxes, cls_scores, iou_threshold)
        # Map back to original indices
        keep_indices.append(
            torch.nonzero(cls_mask, as_tuple=False).squeeze(1)[cls_indices]
        )

    keep_indices = torch.cat(keep_indices)
    sorted_idx = scores[keep_indices].argsort(descending=True)[:max_detections]
    final_indices = keep_indices[sorted_idx]

    final_boxes = boxes[final_indices].cpu()
    final_scores = scores[final_indices].cpu()
    final_labels = labels[final_indices].cpu().long()

    return {
        "boxes": final_boxes,
        "scores": final_scores,
        "labels": final_labels,
    }


def process_evaluation_results(metrics: dict[str, Tensor]) -> dict[str, Any]:
    """
    Process evaluation results from Mean Average Precision metric.

    Args:
        metrics (dict): A dictionary containing evaluation metrics.

    Returns:
        dict: A dictionary with processed 'map_50' and 'map' as map_50_95,
    """
    return {
        "map_50": metrics["map_50"].item(),
        "map_50_95": metrics["map"].item(),
        "classes": metrics["classes"].tolist() if "classes" in metrics else [],
    }
