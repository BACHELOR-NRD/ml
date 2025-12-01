from copy import deepcopy
import math
from typing import List, Literal, Optional, TypedDict

import torch
import numpy as np

from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import (
    convert_pred2eval,
    postprocess_prediction_nms,
    weighted_boxes_fusion,
)
from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_PREDICTION,
)


logger = setup_logger(__name__)


class ScoreDistribution(TypedDict):
    minimum: float
    maximum: float
    mean: float
    std: float
    quantiles: List[float]


def calculate_score_distribution(
    adapter_predictions: list[ADAPTER_PREDICTION],
) -> ScoreDistribution:
    """
    Calculate score distributions (min, max, mean, std) for each adapter's predictions.
    Args:
        adapters_predictions: Per-adapter predictions organized by image.
    Returns:
        dict: Score distributions containing minimums, maximums, means, and stds for each adapter.
    """

    distribution: ScoreDistribution = {
        "minimum": np.nan,
        "maximum": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "quantiles": [],
    }
    all_scores: List[float] = []
    for p in adapter_predictions:
        for score in p["scores"]:
            all_scores.append(score.unsqueeze(0).detach().cpu().item())
    if len(all_scores) == 0:
        logger.error("No predictions; skipping score distribution calculation.")
        raise ValueError("Cannot calculate score distributions with no predictions.")

    all_scores_tensor = torch.tensor(all_scores)
    distribution["minimum"] = all_scores_tensor.min().item()
    distribution["maximum"] = all_scores_tensor.max().item()
    distribution["mean"] = all_scores_tensor.mean().item()
    distribution["std"] = all_scores_tensor.std().item()
    distribution["quantiles"] = all_scores_tensor.quantile(
        torch.linspace(0, 1, steps=101)
    ).tolist()

    return distribution


def normalize_scores(
    preds_list: list[list[torch.Tensor]],
    method: Literal["minmax", "zscore", "quantile"] = "minmax",
    distributions: Optional[List[ScoreDistribution]] = None,
) -> list[list[torch.Tensor]]:
    """
    Normalize confidence scores of predictions from multiple adapters.
    Args:
        preds_list (list): List of predictions from different adapters. Each prediction is a list of tensors per image.
        method (str): Normalization method, either "minmax", "zscore", or "quantile".
        trust (list, optional): Trust weights for each adapter. If None, equal weights are used.
    Returns:
        list: Normalized predictions with adjusted confidence scores.
    """

    normalized_all = []

    for adapter_idx, preds in enumerate(preds_list):
        valid_scores = [p[:, 4] for p in preds if p.numel() > 0]

        if not valid_scores:
            logger.debug(
                "Adapter %d produced no predictions; skipping score normalization.",
                adapter_idx,
            )
            normalized_all.append(deepcopy(preds))
            continue

        flat_scores = torch.cat(valid_scores, dim=0)

        if distributions is None:
            s_min, s_max = flat_scores.min(), flat_scores.max()
            mean, std = flat_scores.mean(), flat_scores.std() + 1e-6
            quantiles = torch.quantile(flat_scores, torch.linspace(0, 1, steps=101))
        else:
            s_min = torch.tensor(distributions[adapter_idx]["minimum"])
            s_max = torch.tensor(distributions[adapter_idx]["maximum"])
            mean = torch.tensor(distributions[adapter_idx]["mean"])
            std = torch.tensor(distributions[adapter_idx]["std"]) + 1e-6
            quantiles = torch.linspace(
                distributions[adapter_idx]["minimum"],
                distributions[adapter_idx]["maximum"],
                steps=101,
            )

        normalized = deepcopy(preds)

        for p in normalized:
            if method == "minmax":
                p[:, 4] = (p[:, 4] - s_min) / (s_max - s_min + 1e-6)
                # NOTE: this needs to be done because of metadata from one distribution
                # being applied to another during ensemble
                p[:, 4] = p[:, 4].clamp(0.0, 1.0)
            elif method == "zscore":
                z = (p[:, 4] - mean) / std
                cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
                p[:, 4] = cdf
            elif method == "quantile":
                # For each score, find its quantile position
                quantized_scores = []
                for score in p[:, 4]:
                    idx = torch.searchsorted(quantiles, score)
                    quantile_value = idx.item() / (len(quantiles) - 1)
                    quantized_scores.append(quantile_value)
                p[:, 4] = torch.tensor(quantized_scores)
            else:
                raise ValueError(f"Unknown normalization method: {method}")

        normalized_all.append(normalized)
    return normalized_all


def scale_scores_with_trust(
    preds_list: list[list[torch.Tensor]],
    trust: list[float],
) -> list[list[torch.Tensor]]:
    """
    tensors are [x1,y1,x2,y2,score,label] and only column 4 gets multiplied
    """

    scaled = []
    for preds, t in zip(preds_list, trust):
        scaled.append(
            [
                (
                    p.clone()
                    if len(p) == 0
                    else p.clone().mul_(torch.tensor([1, 1, 1, 1, t, 1]))
                )
                for p in preds
            ]
        )
    return scaled


def scale_scores_with_exponents(
    preds_list: list[list[torch.Tensor]],
    exponents: list[float],
) -> list[list[torch.Tensor]]:
    """
    preds_list: list over adapters → list over images → tensor[N,6]
    exponents: list of floats, one per adapter
    tensor format: [x1,y1,x2,y2,score,label]
    applies: score = score ** exponent_i
    """
    if len(preds_list) != len(exponents):
        raise ValueError(
            f"len(preds_list)={len(preds_list)} does not match len(exponents)={len(exponents)}"
        )

    scaled = []
    for preds, exp in zip(preds_list, exponents):
        if exp == 1.0:
            # Fast path — no change
            scaled.append([p.clone() for p in preds])
            continue

        per_adapter_scaled = []
        for p in preds:
            if len(p) == 0:
                per_adapter_scaled.append(p.clone())
                continue

            p2 = p.clone()
            p2[:, 4] = p2[:, 4].pow(exp)
            per_adapter_scaled.append(p2)

        scaled.append(per_adapter_scaled)

    return scaled


def fuse_adapters_predictions(
    adapters_predictions: list[list[ADAPTER_PREDICTION]],
    max_detections: int,
    iou_threshold: float,
    conf_threshold: float,
    strategy: Optional[Literal["nms", "wbf"]] = None,
    trust_factors: Optional[list[float]] = None,
    exponent_factors: Optional[list[float]] = None,
    score_normalization_method: Optional[
        Literal["minmax", "zscore", "quantile"]
    ] = None,
    distributions: Optional[List[ScoreDistribution]] = None,
) -> list[ADAPTER_PREDICTION]:
    """
    Fuse per-image predictions from multiple adapters into a single list of ADAPTER_PREDICTIONs.
    Args:
        adapters_predictions: Per-adapter predictions organized by image.
        max_detections: Max boxes to keep per image after fusion.
        iou_threshold: IoU used by NMS/WBF.
        conf_threshold: Confidence floor applied during/after fusion.
        strategy: Fusion backend to apply (NMS, WBF, or simple stacking).
        apply_score_normalization: Whether to normalize scores/trust-scale before fusion.
        score_normalization_method: Normalization scheme passed to normalize_scores.
        trust_weights: Optional per-adapter multipliers applied during normalization.
    """

    num_images = len(adapters_predictions[0])
    list_of_tensors_per_adapter_org = []
    for preds_per_adapter in adapters_predictions:
        per_adapter_tensors: list[torch.Tensor] = []
        for p in preds_per_adapter:
            if len(p["boxes"]) > 0:
                # NOTE: Move tensors to CPU to avoid device mismatch when adapters run on GPU.
                boxes = p["boxes"].detach().cpu()
                scores = p["scores"].detach().cpu().unsqueeze(1)
                labels = p["labels"].detach().cpu().unsqueeze(1).float()
                tensor = torch.cat([boxes, scores, labels], dim=1)
            else:
                tensor = torch.empty((0, 6))
            per_adapter_tensors.append(tensor)
        list_of_tensors_per_adapter_org.append(per_adapter_tensors)

    tensors_for_fusion = list_of_tensors_per_adapter_org
    if score_normalization_method is not None:
        tensors_for_fusion = normalize_scores(
            tensors_for_fusion,
            method=score_normalization_method,
            distributions=distributions,
        )

    if trust_factors is not None:
        if len(trust_factors) != len(tensors_for_fusion):
            raise ValueError(
                f"trust_factors length {len(trust_factors)}!= num adapters {len(tensors_for_fusion)}"
            )
        tensors_for_fusion = scale_scores_with_trust(tensors_for_fusion, trust_factors)

    if exponent_factors is not None:
        if len(exponent_factors) != len(tensors_for_fusion):
            raise ValueError(
                f"exponent_factors length {len(exponent_factors)}!= num adapters {len(tensors_for_fusion)}"
            )
        tensors_for_fusion = scale_scores_with_exponents(
            tensors_for_fusion, exponent_factors
        )

    combined_list_of_tensors = []
    for img_idx in range(num_images):
        combined = torch.cat(
            [
                tensors_for_fusion[adapter_i][img_idx]
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
                nms_combined = postprocess_prediction_nms(
                    boxes=pred_boxes,
                    scores=pred_scores,
                    labels=pred_labels,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold,
                    max_detections=max_detections,
                )
            strategy_predictions.append(nms_combined)
    elif strategy == "wbf":
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
    elif strategy is None:
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

    processed_predictions = [convert_pred2eval(pred) for pred in strategy_predictions]
    return processed_predictions
