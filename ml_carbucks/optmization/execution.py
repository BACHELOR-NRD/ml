import math
import json
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import optuna

from ml_carbucks.optmization.EarlyStoppingCallback import create_early_stopping_callback
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)

OPTUNA_DIRECTION_TYPE = Union[
    List[Literal["maximize", "minimize"]], Literal["maximize", "minimize"]
]


def params_equal(p1, p2, tol=1e-8):
    if p1.keys() != p2.keys():
        return False
    for k in p1:
        v1, v2 = p1[k], p2[k]
        if isinstance(v1, float) and isinstance(v2, float):
            if not math.isclose(v1, v2, rel_tol=tol, abs_tol=tol):
                return False
        else:
            if v1 != v2:
                return False
    return True


def get_existing_trials_info_multiobj(
    trials: List[optuna.trial.FrozenTrial],
    min_percentage_improvement: float,
    directions: OPTUNA_DIRECTION_TYPE,
) -> Tuple[List[int], List[Optional[float]]]:

    temp_directions = directions if isinstance(directions, list) else [directions]

    n_objectives = len(temp_directions)
    no_improvement_counts = [0 for _ in range(n_objectives)]
    best_values = [None for _ in range(n_objectives)]

    for trial in trials:

        if trial.state in (
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ):
            for i in range(n_objectives):
                no_improvement_counts[i] += 1
        elif trial.state == optuna.trial.TrialState.COMPLETE and trial.values is None:
            logger.error(
                f"Trial {trial.number} has no values, but is marked as COMPLETE"
            )
            continue
        elif (
            trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
        ):
            for i, temp_direction in enumerate(temp_directions):
                value = trial.values[i]
                best_value = best_values[i]
                if (
                    best_value is None
                    or (
                        best_value >= 0
                        and temp_direction == "maximize"
                        and value > best_value * (1.0 + min_percentage_improvement)
                    )
                    or (
                        best_value >= 0
                        and temp_direction == "minimize"
                        and value < best_value * (1.0 - min_percentage_improvement)
                    )
                    or (
                        best_value < 0
                        and temp_direction == "maximize"
                        and value > best_value * (1.0 - min_percentage_improvement)
                    )
                    or (
                        best_value < 0
                        and temp_direction == "minimize"
                        and value < best_value * (1.0 + min_percentage_improvement)
                    )
                ):
                    best_values[i] = value
                    no_improvement_counts[i] = 0
                else:
                    no_improvement_counts[i] += 1

    return no_improvement_counts, best_values  # type: ignore


def execute_simple_study(
    hyper_name: str,
    study_name: str,
    results_dir: Path,
    n_trials: int,
    create_objective_func: Callable,
    patience: int = -1,
    min_percentage_improvement: float = 0.01,
    optimization_timeout: Optional[float] = None,
    metadata: Optional[dict] = None,
    append_trials: Optional[list[dict]] = None,
    hyper_suffix: str = "hyper",
    study_attributes: Optional[dict] = None,
    n_jobs: int = 1,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> dict[str, Any]:

    if study_attributes is None:
        study_attributes = {}

    if metadata is None:
        metadata = {}
    else:
        # NOTE: it needs to be json serializable so we need to check for it
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool, list, dict)):
                logger.warning(
                    f"Metadata key '{key}' has non-serializable value: {value}"
                )

    metadata.update(
        {
            "n_jobs": n_jobs,
            "hyper_name": hyper_name,
            "study_name": study_name,
            "trials_planned": n_trials,
            "patience": patience,
            "min_percentage_improvement": min_percentage_improvement,
            "optimization_timeout": optimization_timeout,
            "results_dir": str(results_dir),
        }
    )

    if append_trials is None:
        append_trials = []

    hyper_dir_path = results_dir / hyper_suffix
    hyper_dir_path.mkdir(parents=True, exist_ok=True)

    sql_path = results_dir / "studies" / f"{study_name}.db"
    sql_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{hyper_name}_{study_name}",
        load_if_exists=True,
        storage=f"sqlite:///{sql_path}",
        sampler=sampler,
    )
    study.set_user_attr("study_attributes", study_attributes)

    no_improvement_count, best_value = get_existing_trials_info_multiobj(
        study.get_trials(),
        min_percentage_improvement,
        directions="maximize",
    )

    early_callback = None
    if patience > 0:
        early_callback = create_early_stopping_callback(
            name=study_name,
            study=study,
            patience=patience,
            min_percentage_improvement=min_percentage_improvement,
        )

    for trial_params in append_trials:
        # NOTE this check is probably not needed since we use skip_if_exists=True (needs verification)
        exists = any(params_equal(t.params, trial_params) for t in study.trials)
        if not exists:
            study.enqueue_trial(trial_params, skip_if_exists=True)

    if not (patience > 0 and any(count >= patience for count in no_improvement_count)):
        study.optimize(
            func=create_objective_func(),
            n_trials=n_trials,
            callbacks=[early_callback] if early_callback else None,
            timeout=optimization_timeout,
            gc_after_trial=True,
            n_jobs=n_jobs,
        )
    else:
        logger.warning(
            f"Early stopping activated before starting optimization for {hyper_name}-{study_name}."
        )
        logger.warning(
            f"No improvement counts: {no_improvement_count} with patience {patience}."
        )

    study_time_sum = sum(
        trial.duration.total_seconds() if trial.duration is not None else 0
        for trial in study.trials
    )

    best_trial = None
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed_trials) == 0:
        logger.error(
            f"No completed trials found for model combination {hyper_name}-{study_name}."
        )

    is_single_objective = study.directions is None or len(study.directions) == 1
    if not is_single_objective:
        # NOTE: ROBUST
        best_trial = max(study.best_trials, key=lambda t: min(t.values))
    else:
        best_trial = study.best_trial

    hyper_results = {
        **metadata,
        "best_params": best_trial.params,
        "best_value": best_trial.value if is_single_objective else best_trial.values[0],
        "best_trial_number": best_trial.number,
        "study_name": study_name,
        "trials_completed": len(completed_trials),
        "total_study_time_seconds": study_time_sum,
    }

    (hyper_dir_path / hyper_name).mkdir(parents=True, exist_ok=True)
    with open(hyper_dir_path / hyper_name / f"results_{study_name}.json", "w") as f:
        json.dump(hyper_results, f, indent=4)

    return hyper_results


def execute_custom_study_trial(
    study_name: str,
    results_dir: Path,
    objective_func: Callable,
    params: dict,
    hyper_name: str = "custom_params",
    hyper_suffix: str = "hyper",
    metadata: Optional[dict] = None,
) -> dict[str, Any]:

    if metadata is None:
        metadata = {}
    else:
        # NOTE: it needs to be json serializable so we need to check for it
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool, list, dict)):
                logger.warning(
                    f"Metadata key '{key}' has non-serializable value: {value}"
                )

    hyper_dir_path = results_dir / hyper_suffix
    hyper_dir_path.mkdir(parents=True, exist_ok=True)

    sql_path = results_dir / "studies" / f"{study_name}.db"
    sql_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{hyper_name}_{study_name}",
        load_if_exists=True,
        storage=f"sqlite:///{sql_path}",
    )

    distributed_params = {}
    dynamic_params = {}

    for key, value in params.items():
        if (isinstance(value, float) or isinstance(value, int)) and not isinstance(
            value, bool
        ):
            distributed_params[key] = value
        else:
            dynamic_params[key] = value

    fixed_distributions = {}
    for key, value in distributed_params.items():
        if isinstance(value, int):
            fixed_distributions[key] = optuna.distributions.IntDistribution(
                value, value
            )
        else:
            fixed_distributions[key] = optuna.distributions.FloatDistribution(
                value, value
            )

    trial = study.ask(fixed_distributions=fixed_distributions)

    trial.params.update(params)

    trial.set_user_attr("params", params)
    trial.set_user_attr("distributed_params", distributed_params)
    trial.set_user_attr("dynamic_params", dynamic_params)
    trial.set_user_attr("metadata", metadata)

    score = objective_func(params=params)

    frozen_trial = study.tell(trial, score)

    hyper_results = {
        **metadata,
        "trial_number": frozen_trial.number,
        "distributed_params": frozen_trial.params,
        "dynamic_params": dynamic_params,
        "params": params,
        "value": score,
        "study_name": study_name,
        "trial_time_seconds": (
            frozen_trial.duration.total_seconds() if frozen_trial.duration else 0
        ),
    }

    (hyper_dir_path / hyper_name).mkdir(parents=True, exist_ok=True)
    with open(
        hyper_dir_path
        / hyper_name
        / f"results_custom_{study_name}_trial_{frozen_trial.number}.json",
        "w",
    ) as f:
        json.dump(hyper_results, f, indent=4)

    return hyper_results
