from matplotlib.pyplot import plot, draw, show
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import numpy as np


def plot_label_thresholds(
    thresholds: dict,
    label: str,
    pfx_labs: list = ["True", "False"],
) -> None:
    """
    Plot quantile and mean |thresholds| corresponding to |label|.

    Parameters
    ----------
    thresholds: required, dict
        Thresholds correesponding to the |label|.
    label: required, str
        Label corresponding to the |thresholds|.
    pfx_labs: optional, list
        Prefix labels (e.g. True, False, Null) to plot.

    Returns
    ------
    None
    """
    fig, ax = plt.subplots(nrows=len(thresholds.keys()), ncols=1, figsize=(10, 10))
    for i, (k, v) in enumerate(thresholds.items()):
        v_ = v.cpu().detach().numpy()
        for j in range(len(pfx_labs)):
            ax[i].plot(v_[j, 0, :], label=pfx_labs[j])
        ax[i].legend()
        ax[i].set_title(f"{k} normalized probability of {label}")

    fig.tight_layout()
    show()

    return

def plot_layerwise_metric_heatmaps(
    metrics: dict, pfx_labs: list = ["True", "False"]
) -> None:
    """
    Plot heatmaps for layerwise |metrics|, across the context positions and layers.

    Parameters
    ----------
    metrics : required, dict
        Metrics to plot.

    Returns
    ------
    None
    """
    n_rows, n_cols = len(metrics.keys()), len(pfx_labs)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, n_rows * 4))
    for i, (key, v) in enumerate(metrics.items()):
        ax_ = ax[i] if n_rows > 1 else ax
        for j in range(n_cols):
            sns.heatmap(v[0][j], vmin=0.0, vmax=1.0, ax=ax_[j], cmap="Reds")
            ax_[j].set_title(f"{pfx_labs[j]} prefix,\n {key}", size=20)
            ax_[j].set_xlabel("position in context")
            ax_[j].set_ylabel("layer depth")

    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    show()

    return

def plot_layerwise_metric_curves(
    metrics: dict,
    layer: int = -math.inf,
    context_pos: int = -math.inf,
    pfx_labs: list = ["True", "False"],
    show_confidence: bool = False,
) -> None:
    """
    Plot curves of layerwise |metrics| for a given |context_pos| across the
    layers or for a given |layer| across the contexts.

    Parameters
    ----------
    metrics: required, dict
        Metrics to plot.
    layer: optional, int
        Layer index.
    context_pos: optional, int
        Position of context.
    pfx_labs: optional, list
        Prefix labels (e.g. True, False, Null) to plot.
    show_confidence: optional, bool
        Parameter to show confidence interval lines.

    Returns
    ------
    None
    """
    n_rows, n_cols = len(metrics.keys()), 1
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, n_rows * 3.5))
    colors = ["blue", "orange", "green", "pink", "brown", "gray"]
    legend = list(pfx_labs)
    for i, (key, v) in enumerate(metrics.items()):
        ax_ = ax[i] if n_rows > 1 else ax
        y_min = 1

        for j in range(len(pfx_labs)):
            if pfx_labs[j] == "Null":
                null_curve = v[0][0, :, 0]
                x_axis = list(range(null_curve.shape[0]))
                ax_.plot(x_axis, null_curve, c="green", label=pfx_labs[j])
                continue

            curve = (
                v[3 * (j // 2)][j % 2, layer]
                if layer != -math.inf
                else v[3 * (j // 2)][j % 2, :, context_pos]
            )
            curve_upper = (
                v[3 * (j // 2) + 1][j % 2, layer]
                if layer != -math.inf
                else v[3 * (j // 2) + 1][j % 2, :, context_pos]
            )
            curve_lower = (
                v[3 * (j // 2) + 2][j % 2, layer]
                if layer != -math.inf
                else v[3 * (j // 2) + 2][j % 2, :, context_pos]
            )

            x_axis = list(range(curve.shape[0]))
            ax_.plot(x_axis, curve, c=colors[j], label=pfx_labs[j])

            y_min = min(y_min, min(curve))
            if show_confidence:
                ax_.plot(x_axis, curve_upper, c=colors[j], linestyle="dotted")
                ax_.plot(x_axis, curve_lower, c=colors[j], linestyle="dotted")
                y_min = min(y_min, min(curve_lower))

        ax_.set_xticks([1] + list(range(5, curve.shape[0], 5)))
        ax_.set_xlabel("layer" if context_pos != -math.inf else "position in context")
        ax_.set_ylim(bottom=y_min - 0.1)
        ax_.set_title(key)
        ax_.legend()

#     if context_pos != -math.inf:
#         fig.suptitle(f"Position: {context_pos}", y=1.03, size=20)
#     elif layer != -math.inf:
#         fig.suptitle(f"Layer: {layer}", y=1.03, size=20)
    fig.tight_layout()
    show()

    return