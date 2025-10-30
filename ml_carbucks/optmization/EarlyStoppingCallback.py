from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import optuna

from ml_carbucks.utils.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class EarlyStoppingCallback:
    """When I wrote the initial version of this class, I knew what was up. Now, rewritten to accommodate multiple objectives, I just hope it works"""

    name: str
    patience: int
    min_percentage_improvement: float = 0.0
    best_value: Optional[Union[float, List[float]]] = None
    no_improvement_count: Optional[Union[int, List[int]]] = None
    directions: Optional[Union[str, List[str]]] = None

    def __post_init__(self):

        # NOTE: Normalize inputs to lists for consistency

        if self.best_value is not None and not isinstance(
            self.best_value, (list, tuple)
        ):
            self.best_value = [self.best_value]

        if self.no_improvement_count is not None and not isinstance(
            self.no_improvement_count, (list, tuple)
        ):
            self.no_improvement_count = [self.no_improvement_count]

        if self.directions is not None and not isinstance(
            self.directions, (list, tuple)
        ):
            self.directions = [self.directions]

        self.is_single_objective = len(self.directions) == 1 and self.directions[0] in [  # type: ignore
            "minimize",
            "maximize",
        ]

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:

        if trial.state in (
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ):
            for i in range(len(self.no_improvement_count)):  # type: ignore
                self.no_improvement_count[i] += 1  # type: ignore

        elif trial.state == optuna.trial.TrialState.COMPLETE:

            if self.is_single_objective:
                current_best_values = [study.best_value]
            else:
                current_best_values = list(study.best_trials[0].values)

            if self.best_value is None or all(v is None for v in self.best_value):  # type: ignore
                self.best_value = current_best_values.copy()
                self.no_improvement_count = [0 for _ in current_best_values]

            # Track improvement per objective
            for i, (direction, cur, best) in enumerate(  # type: ignore
                zip(self.directions, current_best_values, self.best_value)  # type: ignore
            ):

                if (
                    (
                        best >= 0
                        and direction == "maximize"
                        and cur > best * (1.0 + self.min_percentage_improvement)
                    )
                    or (
                        best >= 0
                        and direction == "minimize"
                        and cur < best * (1.0 - self.min_percentage_improvement)
                    )
                    or (
                        best < 0
                        and direction == "maximize"
                        and cur > best * (1.0 - self.min_percentage_improvement)
                    )
                    or (
                        best < 0
                        and direction == "minimize"
                        and cur < best * (1.0 + self.min_percentage_improvement)
                    )
                ):
                    self.best_value[i] = cur  # type: ignore
                    self.no_improvement_count[i] = 0  # type: ignore
                else:
                    self.no_improvement_count[i] += 1  # type: ignore

        # Stop if any objective exceeds patience
        if any(c >= self.patience for c in self.no_improvement_count):  # type: ignore
            logger.warning(
                f"Early stopping the study: {self.name} due to "
                + f"no {self.min_percentage_improvement * 100}% improvement for "
                + f"{self.patience} trials | on trial: {trial.number}"
                + f" | best values: {self.best_value} | no improvement counts: {self.no_improvement_count}"
            )
            study.stop()


def get_existing_trials_info_multiobj(
    trials: List[optuna.trial.FrozenTrial],
    min_percentage_improvement: float,
    directions: Union[
        Literal["minimize", "maximize"], List[Literal["minimize", "maximize"]]
    ],
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


def create_early_stopping_callback(
    name: str,
    study: optuna.Study,
    patience: int,
    min_percentage_improvement: float = 0.0,
) -> EarlyStoppingCallback:

    directions: Union[
        Literal["minimize", "maximize"], List[Literal["minimize", "maximize"]]
    ]

    if study.direction == optuna.study.StudyDirection.MINIMIZE:
        directions = "minimize"
    elif study.direction == optuna.study.StudyDirection.MAXIMIZE:
        directions = "maximize"
    else:
        directions = [
            "minimize" if d == optuna.study.StudyDirection.MINIMIZE else "maximize"
            for d in study.directions
        ]

    no_improvement_counts, best_values = get_existing_trials_info_multiobj(
        trials=study.trials,
        min_percentage_improvement=min_percentage_improvement,
        directions=directions,
    )

    return EarlyStoppingCallback(
        name=name,
        patience=patience,
        min_percentage_improvement=min_percentage_improvement,
        best_value=best_values,  # type: ignore
        no_improvement_count=no_improvement_counts,
        directions=directions,  # type: ignore
    )
