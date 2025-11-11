import pandas as pd
from ml_carbucks import RESULTS_DIR
from ml_carbucks.ensemble.EnsembleModel import EnsembleModel


if __name__ == "__main__":
    hyper_runtime = "20251111_165438"
    aggregated_results_path = (
        RESULTS_DIR / "optuna" / f"aggregated_hyper_{hyper_runtime}.csv"
    )

    df_aggregated = pd.read_csv(aggregated_results_path)

    # NOTE: Not implemented yet
    # adapters_list = []
    # for _, row in df_aggregated.iterrows():
    # ensemble = EnsembleModel(
    #     classes=classes,
    #     adapters=adapters_list,
    # )
