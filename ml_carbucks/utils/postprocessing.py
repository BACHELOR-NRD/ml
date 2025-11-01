from typing import List, Literal, Optional

import torch

from torch import Tensor

# -------------------- IoU Computation --------------------


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute IoU between two sets of boxes in [x1, y1, x2, y2]."""

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])

    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])

    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-6)


# -------------------- Non-Max Suppression --------------------


def nms_fusion(preds: Tensor, iou_thresh: float = 0.5) -> Tensor:
    """

    Perform Non-Max Suppression per class.

    Args:

    preds: [K,6] -> [x1, y1, x2, y2, score, class_id]

    Returns:

    [K',6] after suppression

    """

    boxes, scores, labels = preds[:, :4], preds[:, 4], preds[:, 5].int()

    keep_boxes = []

    for c in labels.unique():

        mask = labels == c

        b, s = boxes[mask], scores[mask]

        idxs = s.argsort(descending=True)

        keep = []

        while idxs.numel() > 0:

            i = idxs[0]

            keep.append(i.item())

            if idxs.numel() == 1:

                break

            ious = box_iou(b[i].unsqueeze(0), b[idxs[1:]])[0]

            idxs = idxs[1:][ious < iou_thresh]

            kept = torch.cat(
                [
                    b[keep],
                    s[keep].unsqueeze(1),
                    torch.full((len(keep), 1), c, device=b.device),
                ],
                dim=1,
            )

            keep_boxes.append(kept)

    return (
        torch.cat(keep_boxes)
        if keep_boxes
        else torch.empty((0, 6), device=preds.device)
    )


# -------------------- Weighted Box Fusion --------------------


def wbf_fusion(preds: Tensor, iou_thresh: float = 0.5) -> Tensor:
    """

    Weighted Box Fusion (WBF) per class.

    Args:

    preds: [K,6] -> [x1, y1, x2, y2, score, class_id]

    Returns:

    [K',6]

    """

    boxes, scores, labels = preds[:, :4], preds[:, 4], preds[:, 5].int()

    merged = []

    for c in labels.unique():

        mask = labels == c

        b, s = boxes[mask], scores[mask]

        if len(b) == 0:

            continue

        idxs = s.argsort(descending=True)

        b, s = b[idxs], s[idxs]

        used = torch.zeros(len(b), dtype=torch.bool)

        fused = []

        for i in range(len(b)):

            if used[i]:

                continue

            ious = box_iou(b[i].unsqueeze(0), b)[0]

            mask_iou = ious >= iou_thresh

            used[mask_iou] = True

            weights = s[mask_iou]

            weighted_box = (b[mask_iou] * weights[:, None]).sum(0) / weights.sum()

            weighted_score = s[mask_iou].mean()

            fused.append(
                torch.cat(
                    [weighted_box, torch.tensor([weighted_score, c], device=b.device)]
                )
            )

        merged.append(torch.stack(fused))

    return torch.cat(merged) if merged else torch.empty((0, 6), device=preds.device)


# -------------------- Score Normalization --------------------


def normalize_scores(
    preds_list: List[Tensor],
    method: Literal["minmax", "zscore"] = "minmax",
    trust: Optional[List[float]] = None,
) -> List[Tensor]:
    """

    Normalize and optionally scale scores per model using trust values.

    Args:

    preds_list: list of [N, K_i, 6] per model

    method: 'minmax' or 'zscore'

    trust: optional list of trust weights (same length as preds_list)

    """

    if trust is None:

        trust = [1.0] * len(preds_list)

    assert len(trust) == len(preds_list), "Trust list must match number of models."

    normalized = []

    for preds, t in zip(preds_list, trust):

        flat_scores = preds[:, :, 4]

        if method == "minmax":

            s_min, s_max = flat_scores.min(), flat_scores.max()

            norm_scores = (flat_scores - s_min) / (s_max - s_min + 1e-6)

        elif method == "zscore":

            mean, std = flat_scores.mean(), flat_scores.std() + 1e-6

            norm_scores = (flat_scores - mean) / std

            norm_scores = (norm_scores - norm_scores.min()) / (
                norm_scores.max() - norm_scores.min() + 1e-6
            )

        else:

            raise ValueError(f"Unknown normalization method: {method}")

        # Apply trust factor (acts like scaling confidence by model reliability)

        preds_clone = preds.clone()

        preds_clone[:, :, 4] = norm_scores * t

        normalized.append(preds_clone)

    return normalized


# -------------------- Single Image Merge --------------------


def merge_single_image(
    preds_list: List[Tensor],
    strategy: Literal["wbf", "nms"] = "wbf",
    iou_thresh: float = 0.5,
    score_thresh: float = 0.001,
) -> Tensor:
    """

    Merge predictions from multiple models for a single image.

    Args:

    preds_list: list of [K_i,6]

    Returns:

    [K',6]

    """

    preds = torch.cat(preds_list, dim=0)

    preds = preds[preds[:, 4] > score_thresh]

    if preds.numel() == 0:

        return torch.empty((0, 6))

    if strategy == "nms":

        return nms_fusion(preds, iou_thresh)

    elif strategy == "wbf":

        return wbf_fusion(preds, iou_thresh)

    else:

        raise ValueError(f"Unknown strategy: {strategy}")


# -------------------- Batch-level Merge --------------------


def merge_model_predictions(
    models_preds: List[Tensor],
    strategy: Literal["wbf", "nms"] = "wbf",
    normalize: Optional[Literal["minmax", "zscore"]] = None,
    trust: Optional[List[float]] = None,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.001,
) -> List[Tensor]:
    """

    Merge predictions from multiple detection models.

    Args:

    models_preds: list of [N, K_i, 6]

    normalize: optional score normalization ('minmax' or 'zscore')

    trust: optional list of trust weights per model

    Returns:

    list of N tensors [K',6]

    """

    if normalize is not None:

        models_preds = normalize_scores(models_preds, method=normalize, trust=trust)

    N = models_preds[0].shape[0]

    merged_results: List[Tensor] = []

    for img_idx in range(N):

        preds_list = [m[img_idx] for m in models_preds]

        merged = merge_single_image(preds_list, strategy, iou_thresh, score_thresh)

        merged_results.append(merged)

    return merged_results
