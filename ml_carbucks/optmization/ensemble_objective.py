from pathlib import Path
from typing import Any, Callable, Dict, List

import pickle as pkl
import optuna
from optuna import Trial
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_DATASETS,
    ADAPTER_METRICS,
    ADAPTER_PREDICTION,
    BaseDetectionAdapter,
)
from ml_carbucks.optmization.TrialParamWrapper import TrialParamWrapper
from ml_carbucks.utils.ensembling import (
    ScoreDistribution,
    calculate_score_distribution,
    fuse_adapters_predictions,
)
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.postprocessing import (
    convert_pred2eval,
    postprocess_evaluation_results,
)
from ml_carbucks.utils.preprocessing import create_clean_loader

logger = setup_logger(__name__)


def create_objective(
    adapters_predictions: List[List[ADAPTER_PREDICTION]],
    ground_truths: List[dict],
    distributions: List[ScoreDistribution],
) -> Callable:
    """
    Ensemble optimization objective function creator.
    """

    def objective(trial: Trial) -> float:
        """
        Ensemble optimization objective function.
        It fuses the predictions of the adapters using the parameters suggested by the trial,
        computes the mean average precision (mAP) of the fused predictions against the ground truths,
        and returns the mAP as the objective value to be maximized.
        """
        try:

            params = TrialParamWrapper(
                ensemble_size=len(adapters_predictions)
            ).get_param(trial, "ensemblemodel")

            fused_predictions = fuse_adapters_predictions(
                adapters_predictions=adapters_predictions,
                max_detections=params["fusion_max_detections"],
                iou_threshold=params["fusion_iou_threshold"],
                conf_threshold=params["fusion_conf_threshold"],
                strategy=params["fusion_strategy"],
                trust_weights=params["fusion_trust_weights"],
                score_normalization_method=params["fusion_norm_method"],
                distributions=distributions,
            )

            processed_predictions = [
                convert_pred2eval(pred) for pred in fused_predictions
            ]

            metric = MeanAveragePrecision()
            metric.update(processed_predictions, ground_truths)  # type: ignore
            processed = postprocess_evaluation_results(metric.compute())

            score = processed["map_50"]

            return score
        except optuna.exceptions.TrialPruned as e:
            logger.error(f"Trial {trial.number} pruned: {e}")
            trial.set_user_attr("error", str(e))
            raise
        except Exception as e:
            logger.error(f"Error in objective: {e}")
            trial.set_user_attr("error", str(e))
            return -0.1

    return objective


def create_ensembling_opt_prestep(
    adapters: List[BaseDetectionAdapter],
    train_folds: List[ADAPTER_DATASETS],
    val_folds: List[ADAPTER_DATASETS],
    runtime: str,
    results_dir: Path,
) -> tuple[
    List[List[ADAPTER_PREDICTION]], List[dict], List[ScoreDistribution], Dict[str, Any]
]:
    """
    This function aims at allowing the ensmble optimization to be faster by precomputing:
    1. predictions of each adapter on validation folds
    2. ground truths for validation folds
    3. score distributions for each adapter's predictions

    Additioanlly, it saves these precomputed results to disk for future runs in case of debugging or re-running the optimization.
    """
    saved_prestep_path = results_dir / "ensemble" / runtime / f"prestep_{runtime}.pkl"
    saved_prestep_path.parent.mkdir(parents=True, exist_ok=True)

    adapters_predictions: List[List[ADAPTER_PREDICTION]] = [
        [] for _ in range(len(adapters))
    ]
    ground_truths: List[dict] = []
    distributions: List[ScoreDistribution] = []
    adapters_crossval_metrics: List[List[ADAPTER_METRICS]] = [
        [] for _ in range(len(adapters))
    ]

    if saved_prestep_path.exists():
        (
            adapters_predictions,
            ground_truths,
            distributions,
            adapters_crossval_metrics,
        ) = pkl.load(open(saved_prestep_path, "rb"))
    else:
        for fold_idx in range(len(train_folds)):
            fold_train_datasets = train_folds[fold_idx]
            fold_val_datasets = val_folds[fold_idx]

            new_adapters = [adapter.clone().setup() for adapter in adapters]
            logger.info(f"Processing fold {fold_idx + 1}/{len(train_folds)}")

            for adapter_idx in range(len(new_adapters)):
                logger.info(
                    f"Fitting adapter {adapter_idx + 1}/{len(new_adapters)} on fold {fold_idx + 1}/{len(train_folds)}"
                )
                new_adapters[adapter_idx] = new_adapters[adapter_idx].fit(
                    datasets=fold_train_datasets
                )

            val_loader = create_clean_loader(
                datasets=fold_val_datasets,
                shuffle=False,
                batch_size=8,
                transforms=None,
            )

            evaluators_list = [MeanAveragePrecision() for _ in range(len(new_adapters))]

            for batch in val_loader:
                images, targets = batch

                for adapter_idx, adapter in enumerate(new_adapters):
                    preds = adapter.predict(images)

                    eval_preds = [
                        {
                            "boxes": pred["boxes"].detach().cpu(),
                            "scores": pred["scores"].detach().cpu(),
                            "labels": pred["labels"].detach().cpu().long(),
                        }
                        for pred in preds
                    ]
                    eval_gts = [
                        {
                            "boxes": target["boxes"].detach().cpu(),
                            "labels": target["labels"].detach().cpu().long(),
                        }
                        for target in targets
                    ]

                    adapters_predictions[adapter_idx].extend(eval_preds)  # type: ignore
                    ground_truths.extend(eval_gts)

                    evaluators_list[adapter_idx].update(eval_preds, eval_gts)

            for adapter_idx, evaluator in enumerate(evaluators_list):
                fold_metrics = postprocess_evaluation_results(evaluator.compute())
                adapters_crossval_metrics[adapter_idx].append(fold_metrics)

        distributions = [
            calculate_score_distribution(preds) for preds in adapters_predictions
        ]

        pkl.dump(
            (
                adapters_predictions,
                ground_truths,
                distributions,
                adapters_crossval_metrics,
            ),
            open(saved_prestep_path, "wb"),
        )

    for adapter_idx, adapters_metrics in enumerate(adapters_crossval_metrics):
        for metric_idx, metrics in enumerate(adapters_metrics):
            logger.info(
                f"Adapter {adapter_idx + 1}/{len(adapters)} - Fold {metric_idx + 1}/{len(adapters_metrics)} - mAP@0.5: {metrics['map_50']:.4f}"
            )

    metadata = {
        "adapters_crossval_metrics": adapters_crossval_metrics,
        "adapters_avg_fold_map_50": [
            sum(fold_metrics["map_50"] for fold_metrics in adapter_metrics)
            / len(adapter_metrics)
            for adapter_metrics in adapters_crossval_metrics
        ],
    }

    return adapters_predictions, ground_truths, distributions, metadata
