from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

import pickle as pkl
import optuna
from optuna import Trial
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ml_carbucks.adapters.BaseDetectionAdapter import (
    ADAPTER_DATASETS,
    ADAPTER_METRICS,
    ADAPTER_PREDICTION,
    BaseDetectionAdapter,
)
from ml_carbucks.ensemble.EnsembleModel import EnsembleModel
from ml_carbucks.optmization.TrialParamWrapper import TrialParamWrapper
from ml_carbucks.utils.ensemble_merging import (
    ScoreDistribution,
    calculate_score_distribution,
    fuse_adapters_predictions,
)
from ml_carbucks.utils.hashing import compute_ensemble_prestep_hash
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
    param_wrapper_version: Literal["v3", "v4", "v5"],
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
                ensemble_size=len(adapters_predictions), version=param_wrapper_version
            ).get_param(trial, "ensemblemodel")

            fused_predictions = fuse_adapters_predictions(
                adapters_predictions=adapters_predictions,
                max_detections=params["fusion_max_detections"],
                iou_threshold=params["fusion_iou_threshold"],
                conf_threshold=params["fusion_conf_threshold"],
                strategy=params["fusion_strategy"],
                trust_factors=params["fusion_trust_factors"],
                exponent_factors=params["fusion_exponent_factors"],
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

    # NOTE: this function is a MESS but it dont want to fix it now

    all_hash = compute_ensemble_prestep_hash(
        adapters=adapters,
        train_folds=train_folds,
        val_folds=val_folds,
        digest_size=8,
    )

    saved_prestep_path = results_dir / "ensemble" / f"prestep_{all_hash}.pkl"

    adapters_predictions: List[List[ADAPTER_PREDICTION]] = [
        [] for _ in range(len(adapters))
    ]
    ground_truths: List[dict] = []
    distributions: List[ScoreDistribution] = []
    adapters_crossval_metrics: List[List[ADAPTER_METRICS]] = [
        [] for _ in range(len(adapters))
    ]
    adapters_dataset_metrics: List[ADAPTER_METRICS] = []

    if saved_prestep_path.exists():
        obj = pkl.load(open(saved_prestep_path, "rb"))
        (
            adapters_predictions,
            ground_truths,
            distributions,
            adapters_crossval_metrics,
            adapters_dataset_metrics,
        ) = obj

    else:
        all_dataset_evaluators = [MeanAveragePrecision() for _ in range(len(adapters))]
        for fold_idx in range(len(train_folds)):
            fold_train_datasets = train_folds[fold_idx]
            fold_val_datasets = val_folds[fold_idx]

            new_adapters = [adapter.clone() for adapter in adapters]

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

            for images, targets in tqdm(val_loader):

                eval_gts = [
                    {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu().long(),
                    }
                    for target in targets
                ]
                ground_truths.extend(eval_gts)

                for adapter_idx, adapter in enumerate(new_adapters):
                    preds = adapter.predict(
                        images,
                        conf_threshold=0.05,
                        max_detections=5,
                        iou_threshold=0.7,
                    )

                    eval_preds = [
                        {
                            "boxes": pred["boxes"].detach().cpu(),
                            "scores": pred["scores"].detach().cpu(),
                            "labels": pred["labels"].detach().cpu().long(),
                        }
                        for pred in preds
                    ]

                    adapters_predictions[adapter_idx].extend(eval_preds)  # type: ignore

                    evaluators_list[adapter_idx].update(eval_preds, eval_gts)
                    all_dataset_evaluators[adapter_idx].update(eval_preds, eval_gts)

            for adapter_idx, evaluator in enumerate(evaluators_list):
                fold_metrics = postprocess_evaluation_results(evaluator.compute())
                adapters_crossval_metrics[adapter_idx].append(fold_metrics)

        distributions = [
            calculate_score_distribution(preds) for preds in adapters_predictions
        ]

        for adapter_idx, evaluator in enumerate(all_dataset_evaluators):
            dataset_metrics = postprocess_evaluation_results(evaluator.compute())
            adapters_dataset_metrics.append(dataset_metrics)

        saved_prestep_path.parent.mkdir(parents=True, exist_ok=True)
        pkl.dump(
            (
                adapters_predictions,
                ground_truths,
                distributions,
                adapters_crossval_metrics,
                adapters_dataset_metrics,
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
        "adapters_dataset_metrics": adapters_dataset_metrics,
        "adapters_dataset_map_50": [
            adapter_metrics["map_50"] for adapter_metrics in adapters_dataset_metrics
        ],
    }

    return adapters_predictions, ground_truths, distributions, metadata


def create_ensemble(
    adapters: List[BaseDetectionAdapter],
    params: Dict[str, Any],
    runtime: str,
    distributions: List[ScoreDistribution],
    results_dir: Path,
    final_datasets: ADAPTER_DATASETS | None = None,
) -> EnsembleModel:
    """
    A function that creates and fits an EnsembleModel from given adapters and parameters.
    The idea is to create a final ensemble model that would be production ready.
    """

    # NOTE: this is quite stupid but necessary to clone,setup and clone without weights again
    # 1. first clone disassociates the object from the original one (not strictly necessary here but good practice)
    # 2. loads all hyperparameters and setups the adapter
    # 3. finally, clone again but this time clean the saved weights to avoid carrying over any trained weights
    ensemble_adapters = [adapter.clone() for adapter in adapters]
    ensemble_params = TrialParamWrapper.convert_ensemble_params_to_model_format(
        params, ensemble_size=len(ensemble_adapters)
    )
    ensemble_model = EnsembleModel(
        **ensemble_params,
        adapters=ensemble_adapters,
        distributions=distributions,
    )
    if final_datasets is None:
        logger.warning(
            "Full datasets for ensemble training not provided. EnsembleModel will be created without fitting."
        )
    else:
        ensemble_model.fit(final_datasets)

    ensemble_model.save(results_dir / "ensemble" / runtime, suffix=runtime)

    return ensemble_model
