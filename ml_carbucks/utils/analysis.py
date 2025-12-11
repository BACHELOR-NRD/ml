from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def plot_analysis_stats_std(
    df,
    target_col: str,
    category_col: str,
    value_col: str = "value",
    title="Analysis X",
):
    df = df.copy().reset_index()

    # Keep the manipulation order
    df[target_col] = pd.Categorical(
        df[target_col], categories=df[target_col].unique().tolist(), ordered=True
    )

    # Compute mean + std
    grouped = df.groupby([target_col, category_col])[value_col]
    stats = grouped.agg(["mean", "std"]).reset_index()

    pivot_mean = stats.pivot(index=target_col, columns=category_col, values="mean")
    pivot_std = stats.pivot(index=target_col, columns=category_col, values="std")
    manipulations = pivot_mean.index.tolist()
    models = pivot_mean.columns.tolist()

    x = np.arange(len(manipulations))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(models))]

    # Plot bars with vertical STD error bars
    for i, (model, color) in enumerate(zip(models, colors)):
        bar_positions = x + i * bar_width
        bar_heights = pivot_mean[model].values
        bar_errors = pivot_std[model].values

        # Draw the bar
        ax.bar(
            bar_positions,
            bar_heights,
            width=bar_width,
            color=color,
            alpha=0.85,
            label=model,
            linewidth=1.5,
        )

        # Draw vertical error bars
        ax.errorbar(
            bar_positions,
            bar_heights,
            yerr=bar_errors,
            fmt="none",
            ecolor="gray",
            elinewidth=1.5,
            capsize=5,
        )

    # Average line across models
    avg_vals = pivot_mean.mean(axis=1).values
    avg_x = x + bar_width * (len(models) - 1) / 2
    ax.plot(
        avg_x,
        avg_vals,
        marker=".",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="black",
        label="Average",
    )

    # Custom legend for STD bars (vertical bar only)
    std_legend = Line2D([0], [0], color="gray", lw=2, label="STD across folds")

    # Combine legend handles
    handles, labels = ax.get_legend_handles_labels()
    handles.append(std_legend)
    labels.append("STD across folds")
    ax.legend(
        handles=handles, labels=labels, frameon=True, framealpha=0.9, loc="upper center"
    )

    # Labels and formatting
    ax.set_xticks(avg_x)
    ax.set_xticklabels(manipulations, rotation=45, ha="right")
    ax.set_ylabel("mAP50")
    ax.set_xlabel("Manipulation Type")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
