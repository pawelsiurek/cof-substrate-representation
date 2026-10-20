"""Plots shared by the notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cofreg.metrics import HIGHER_IS_BETTER

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LABELS = {
    "mape": "MAPE",
    "pa": "Pairwise acc.",
    "rmse": "RMSE",
    "male": "MALE",
    "mape_ge20": "MAPE (y ≥ 20)",
    "smape": "SMAPE",
    "mae": "MAE",
    "r2": "R²",
}


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def metric_dots(summary: pd.DataFrame, metrics=("mape", "pa", "rmse", "male"), highlight: dict | None = None, label_col: str = "model"):
    """One panel per metric: mean ± std over CV repeats, one row per run, best value marked.

    highlight: {model_name: color} for runs to colour (others are blue); baselines are usually grey.
    """
    style()
    highlight = highlight or {}
    names = summary[label_col].tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 0.32 * len(names) + 1.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        mean, std = summary[f"{m}_mean"].to_numpy(), summary[f"{m}_std"].to_numpy()
        ypos = np.arange(len(names))[::-1]
        colors = [highlight.get(n, BLUE) for n in names]
        ax.errorbar(mean, ypos, xerr=std, fmt="none", ecolor=GRAY, elinewidth=1.2, capsize=0)
        ax.scatter(mean, ypos, c=colors, s=36, zorder=3, edgecolors="white", linewidths=1)
        best = np.argmax(mean) if HIGHER_IS_BETTER[m] else np.argmin(mean)
        ax.scatter([mean[best]], [ypos[best]], s=110, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
        arrow = "higher is better" if HIGHER_IS_BETTER[m] else "lower is better"
        ax.set_title(LABELS[m], loc="left")
        ax.set_xlabel(arrow)
        ax.set_yticks(ypos, names)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def pred_vs_true(y: np.ndarray, oof: np.ndarray, title: str, ax=None):
    """Mean OOF prediction over repeats vs truth, log-log, with the repeat range as a bar."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.0))
    mean = np.exp(np.log(np.maximum(oof, 1e-3)).mean(0))
    lo, hi = oof.min(0), oof.max(0)
    ax.vlines(y, lo, hi, color=GRAY, linewidth=1, alpha=0.7)
    ax.scatter(y, mean, s=22, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3)
    lim = (1.5, 1000)
    ax.plot(lim, lim, color=INK_2, linewidth=1, linestyle="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="true capacity", ylabel="predicted (OOF, mean of repeats)")
    ax.set_title(title, loc="left")
    return ax


def ape_contributions(y: np.ndarray, pred: np.ndarray, ax=None, label: str = ""):
    """Share of total MAPE contributed by each sample, sorted by true capacity."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.8))
    order = np.argsort(y)
    ape = np.abs(pred - y) / y
    ax.bar(np.arange(len(y)), ape[order] / ape.sum() * 100, color=BLUE, width=0.8)
    ax.set(xlabel="substrates sorted by true capacity", ylabel="% of total MAPE")
    ax.set_title(label, loc="left")
    ax.set_xticks([])
    return ax
