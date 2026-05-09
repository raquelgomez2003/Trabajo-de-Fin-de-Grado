"""
plotter.py
All matplotlib figures for BioVisor.

Functions
---------
plot_signals_with_stress(signals, fs_map, intervals, stress_map)
    → Figure   (viewer window — signals + stress overlay)

plot_boxplots(features_per_phase, signal_name)
    → Figure   (analysis window — boxplots per feature per phase)

plot_roc(y_true, y_scores, signal_name)
    → Figure   (analysis window — ROC curve + AUC)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve, auc

from app.core.config import PHASE_COLORS
from app.core.models import FEATURE_NAMES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _shade_phases(ax: plt.Axes, intervals: dict[str, list[tuple[float, float]]]) -> None:
    """Shade phase intervals on *ax*. intervals = {phase_key: [(start, end), ...]}"""
    for phase_key, spans in intervals.items():
        color, _ = PHASE_COLORS.get(phase_key, ("#eeeeee", phase_key))
        for start, end in spans:
            ax.axvspan(start, end, color=color, alpha=0.25, zorder=0)


def _stress_overlay(ax: plt.Axes, t_centers: np.ndarray,
                    predictions: np.ndarray, y_max: float) -> None:
    """Draw red markers above the signal wherever stress is predicted."""
    stressed = t_centers[predictions == 1]
    if len(stressed):
        ax.scatter(stressed, np.full(len(stressed), y_max),
                   color="#cc0000", marker="v", s=30, zorder=5,
                   label="Predicted stress")


# ── Viewer: multi-signal + stress markers ─────────────────────────────────────

def plot_signals_with_stress(
    signals: dict[str, np.ndarray],
    fs_map: dict[str, float],
    phase_intervals: dict[str, list[tuple[float, float]]],
    stress_map: dict[str, tuple[np.ndarray, np.ndarray]],  # {sig: (t_centers, preds)}
    bpm_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> Figure:
    """
    One subplot per signal.
    Shades phase regions, marks predicted-stress windows with red triangles.
    If bpm_data is provided and ECG is present, overlays BPM on ECG panel.
    """
    sig_names = list(signals.keys())
    n = len(sig_names)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No signals loaded", ha="center", va="center")
        return fig

    fig, axs = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=True)
    if n == 1:
        axs = [axs]

    fig.suptitle("Signal Viewer", fontsize=13, fontweight="bold", y=0.995)

    for ax, name in zip(axs, sig_names):
        sig  = signals[name]
        fs   = fs_map.get(name, 2000)
        data = sig[:, 0] if sig.ndim == 2 else sig
        t    = np.arange(len(data)) / fs

        ax.plot(t, data, lw=0.7, color="#2255aa", label=name)
        _shade_phases(ax, phase_intervals)

        # Stress overlay
        if name in stress_map:
            t_c, preds = stress_map[name]
            _stress_overlay(ax, t_c, preds, float(np.nanmax(data)))

        # BPM twin axis on ECG
        if name == "ECG" and bpm_data is not None:
            t_bpm, bpm = bpm_data
            if len(t_bpm):
                ax2 = ax.twinx()
                ax2.plot(t_bpm, bpm, color="#e06000", lw=1.2,
                         linestyle="--", label="BPM")
                ax2.set_ylabel("BPM", color="#e06000", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="#e06000")

        ax.set_ylabel(name, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(name, fontsize=9, loc="left")

    axs[-1].set_xlabel("Time (s)", fontsize=9)

    # Legend for phases
    legend_patches = [
        mpatches.Patch(color=col, alpha=0.5, label=lbl)
        for col, lbl in PHASE_COLORS.values()
    ]
    legend_patches.append(
        plt.Line2D([0], [0], marker="v", color="w",
                   markerfacecolor="#cc0000", markersize=7, label="Predicted stress")
    )
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    return fig


# ── Analysis: Boxplots ─────────────────────────────────────────────────────────

def plot_boxplots(
    phase_features: dict[str, np.ndarray],   # {phase_label: X [n_windows, n_features]}
    signal_name: str,
    feature_names: list[str] = FEATURE_NAMES,
) -> Figure:
    """
    Grid of boxplots: one subplot per feature, one box per phase.
    """
    n_feat = len(feature_names)
    ncols  = 4
    nrows  = (n_feat + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(4 * ncols, 3.5 * nrows))
    axs = axs.flatten()
    fig.suptitle(f"Feature Distribution — {signal_name}",
                 fontsize=12, fontweight="bold")

    phase_labels = list(phase_features.keys())
    colors = ["#5599dd", "#55bb77", "#8877dd", "#dd5555"]

    for fi, feat_name in enumerate(feature_names):
        ax = axs[fi]
        data_per_phase = []
        for phase in phase_labels:
            X = phase_features[phase]
            if X.ndim == 2 and X.shape[1] > fi:
                data_per_phase.append(X[:, fi])
            else:
                data_per_phase.append(np.array([]))

        bp = ax.boxplot(data_per_phase, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, col in zip(bp["boxes"], colors[:len(phase_labels)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(phase_labels) + 1))
        ax.set_xticklabels(phase_labels, fontsize=7, rotation=15)
        ax.set_title(feat_name, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused subplots
    for ax in axs[n_feat:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# ── Analysis: ROC curve ────────────────────────────────────────────────────────

def plot_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    signal_name: str,
) -> Figure:
    """
    ROC curve with AUC annotation.
    y_scores: probability of stress class (float in [0, 1]).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2255aa", lw=2,
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random classifier")

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx],
               color="#cc0000", zorder=5, s=60,
               label=f"Best threshold = {thresholds[best_idx]:.3f}")

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve — {signal_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
