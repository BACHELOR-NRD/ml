import math
import json
from pathlib import Path
from typing import Callable, Optional

import optuna

from ml_carbucks.optmization.EarlyStoppingCallback import create_early_stopping_callback
from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


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
):
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
            "hyper_name": hyper_name,
            "study_name": study_name,
            "n_trials": n_trials,
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
        study_name=f"{study_name}_{hyper_name}",
        load_if_exists=True,
        storage=f"sqlite:///{sql_path}",
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
        exists = any(params_equal(t.params, trial_params) for t in study.trials)
        if not exists:
            study.enqueue_trial(trial_params)

    study.optimize(
        func=create_objective_func(),
        n_trials=n_trials,
        callbacks=[early_callback] if early_callback else None,
        timeout=optimization_timeout,
        gc_after_trial=True,
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
        "best_params": best_trial.params,
        "best_value": best_trial.value if is_single_objective else best_trial.values[0],
        "best_trial_number": best_trial.number,
        "study_name": study_name,
        "n_trials": len(completed_trials),
        "total_study_time_seconds": study_time_sum,
        **metadata,
    }

    (hyper_dir_path / hyper_name).mkdir(parents=True, exist_ok=True)
    with open(hyper_dir_path / hyper_name / f"results_{study_name}.json", "w") as f:
        json.dump(hyper_results, f)

    return hyper_results
