"""
plotter.py
All matplotlib figures for BioVisor.

Public API
----------
plot_signals_with_stress(signals, fs_map, phase_intervals, stress_map, bpm_data, rr_data)
plot_boxplots(phase_features, signal_name)
plot_roc(y_true, y_scores, signal_name)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve, auc

from app.core.config import PHASE_COLORS

FEATURE_NAMES = ["Mean", "Std", "RMS", "PtP"]
FEATURE_DIM   = 4


# ── Shared helpers ────────────────────────────────────────────────────────────

def _shade_phases(ax: plt.Axes, intervals: dict) -> None:
    for phase_key, spans in intervals.items():
        color, _ = PHASE_COLORS.get(phase_key, ("#eeeeee", phase_key))
        for start, end in spans:
            ax.axvspan(start, end, color=color, alpha=0.22, zorder=0)


def _stress_markers(ax: plt.Axes, t_centers: np.ndarray,
                    predictions: np.ndarray, y_pos: float) -> None:
    stressed = t_centers[predictions == 1]
    if len(stressed):
        ax.scatter(stressed, np.full(len(stressed), y_pos),
                   color="#cc0000", marker="v", s=28, zorder=6,
                   label="Predicted stress ▼")


# ── BPM panel ─────────────────────────────────────────────────────────────────

def _draw_bpm_panel(ax: plt.Axes, t_bpm: np.ndarray,
                    bpm: np.ndarray, phase_intervals: dict) -> None:
    ax.step(t_bpm, bpm, where="post", color="#e06000", lw=1.5, zorder=3)
    ax.fill_between(t_bpm, bpm, alpha=0.10, color="#e06000", step="post")

    bpm_mean = np.mean(bpm)
    thresh   = bpm_mean + np.std(bpm)

    ax.axhline(bpm_mean, color="#777777", lw=0.9, ls="--",
               label=f"Mean {bpm_mean:.0f} bpm")
    ax.axhline(thresh,   color="#cc0000", lw=0.9, ls=":",
               label=f"Stress threshold {thresh:.0f} bpm")

    # Shade windows above threshold
    for i in range(len(t_bpm) - 1):
        if bpm[i] > thresh:
            ax.axvspan(t_bpm[i], t_bpm[i + 1], color="#ffcccc", alpha=0.45, zorder=0)
    if len(t_bpm) > 1 and bpm[-1] > thresh:
        ax.axvspan(t_bpm[-1], t_bpm[-1] + (t_bpm[-1] - t_bpm[-2]),
                   color="#ffcccc", alpha=0.45, zorder=0)

    # Flecha roja en puntos donde el BPM sube bruscamente y está en zona de estrés
    diffs = np.diff(bpm)
    for ji in np.where(np.abs(diffs) > 3)[0]:
        if bpm[ji] > thresh:
            ax.annotate("",
                        xy=(t_bpm[ji], bpm[ji]),
                        xytext=(t_bpm[ji], bpm[ji] + 2),
                        arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.2),
                        annotation_clip=True)

    _shade_phases(ax, phase_intervals)
    ax.set_ylabel("BPM", fontsize=8)
    ax.set_title("Heart Rate (BPM) — red zone: above personal stress threshold",
                 fontsize=8, loc="left", pad=2)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.22)


# ── RR interval panel ─────────────────────────────────────────────────────────

def _draw_rr_panel(ax: plt.Axes, rr_times: np.ndarray,
                   rr_ms: np.ndarray, phase_intervals: dict) -> None:
    ax.plot(rr_times, rr_ms, color="#9933aa", lw=0.8, zorder=3, label="RR interval")

    rr_mean    = np.mean(rr_ms)
    low_thresh = rr_mean - np.std(rr_ms)

    ax.axhline(rr_mean,    color="#777777", lw=0.9, ls="--",
               label=f"Mean {rr_mean:.0f} ms")
    ax.axhline(low_thresh, color="#cc0000", lw=0.9, ls=":",
               label=f"Stress threshold {low_thresh:.0f} ms")

    mask = rr_ms < low_thresh
    if mask.any():
        ax.scatter(rr_times[mask], rr_ms[mask],
                   color="#cc0000", s=10, zorder=5, label="Short RR (stress)")

    # Flechitas en caídas bruscas de RR (aceleración súbita)
    drops = np.where(np.diff(rr_ms) < -30)[0]
    for di in drops:
        ax.annotate("",
                    xy=(rr_times[di + 1], rr_ms[di + 1]),
                    xytext=(rr_times[di + 1], rr_ms[di + 1] + 25),
                    arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.0),
                    annotation_clip=True)

    _shade_phases(ax, phase_intervals)
    ax.set_ylabel("RR (ms)", fontsize=8)
    ax.set_title("RR Interval — red dots: short RR (stress)  ↓ arrows: sudden acceleration",
                 fontsize=8, loc="left", pad=2)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.22)
    ax.invert_yaxis()


# ── Main viewer figure ────────────────────────────────────────────────────────

def plot_signals_with_stress(
    signals: dict[str, np.ndarray],
    fs_map: dict[str, float],
    phase_intervals: dict,
    stress_map: dict,
    bpm_data: tuple[np.ndarray, np.ndarray] | None = None,
    rr_data:  tuple[np.ndarray, np.ndarray] | None = None,
) -> Figure:
    if not signals:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No signals loaded", ha="center", va="center")
        return fig

    sig_names = list(signals.keys())
    has_bpm   = bpm_data is not None and len(bpm_data[0]) > 0
    has_rr    = rr_data  is not None and len(rr_data[0])  > 0

    panels = [(name, 2.2 if name == "ECG" else 1.0) for name in sig_names]
    if has_bpm:
        panels.append(("__BPM__", 1.4))
    if has_rr:
        panels.append(("__RR__",  1.4))

    height_ratios = [p[1] for p in panels]
    fig, axs = plt.subplots(
        len(panels), 1,
        figsize=(13, max(sum(height_ratios) * 1.7, 6)),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    if len(panels) == 1:
        axs = [axs]

    fig.suptitle("Signal Viewer", fontsize=13, fontweight="bold", y=0.999)

    for ax, (label, _) in zip(axs, panels):
        if label == "__BPM__":
            _draw_bpm_panel(ax, bpm_data[0], bpm_data[1], phase_intervals)
            continue
        if label == "__RR__":
            _draw_rr_panel(ax, rr_data[0], rr_data[1], phase_intervals)
            continue

        sig  = signals[label]
        fs   = fs_map.get(label, 2000)
        data = sig.ravel()

        # Downsample para display (máximo 10.000 puntos)
        step = max(1, len(data) // 10_000)
        t    = np.arange(len(data))[::step] / fs
        ax.plot(t, data[::step], lw=0.55, color="#2255aa", rasterized=True)
        _shade_phases(ax, phase_intervals)

        if label in stress_map:
            t_c, preds = stress_map[label]
            _stress_markers(ax, t_c, preds, float(np.nanpercentile(data, 99)))

        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9, loc="left", pad=2)
        ax.grid(True, alpha=0.22)

    axs[-1].set_xlabel("Time (s)", fontsize=9)

    # Leyenda — sin Baseline, solo fases relevantes + marcador de estrés
    patches = [
        mpatches.Patch(color=col, alpha=0.5, label=lbl)
        for key, (col, lbl) in PHASE_COLORS.items()
        if key != "baseline"
    ]
    patches.append(plt.Line2D([0], [0], marker="v", color="w",
                               markerfacecolor="#cc0000", markersize=7,
                               label="Predicted stress ▼"))
    patches.append(plt.Line2D([0], [0], color="#cc0000", lw=1.5,
                               marker=">", markersize=6,
                               markerfacecolor="#cc0000",
                               label="BPM stress point →"))
    fig.legend(handles=patches, loc="lower center",
               ncol=len(patches), fontsize=8, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=[0, 0.03, 1, 0.998])
    plt.subplots_adjust(hspace=0.5)
    return fig


# ── Boxplots ──────────────────────────────────────────────────────────────────

def plot_boxplots(phase_features: dict[str, np.ndarray], signal_name: str) -> Figure:
    phase_labels = list(phase_features.keys())
    colors       = ["#5599dd", "#55bb77", "#8877dd", "#dd5555"]

    fig, axs = plt.subplots(1, FEATURE_DIM, figsize=(4 * FEATURE_DIM, 4))
    fig.suptitle(f"Feature Distribution — {signal_name}",
                 fontsize=12, fontweight="bold")

    for fi, feat_name in enumerate(FEATURE_NAMES):
        ax = axs[fi]
        data_per_phase = [
            phase_features[ph][:, fi]
            if phase_features[ph].ndim == 2 and phase_features[ph].shape[1] > fi
            else np.array([])
            for ph in phase_labels
        ]
        bp = ax.boxplot(data_per_phase, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, col in zip(bp["boxes"], colors[:len(phase_labels)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        # Limitar eje Y al percentil 2-98 para ignorar outliers extremos
        all_vals = np.concatenate([d for d in data_per_phase if len(d) > 0])
        if len(all_vals) > 0:
            ax.set_ylim(np.percentile(all_vals, 2), np.percentile(all_vals, 98))

        ax.set_xticks(range(1, len(phase_labels) + 1))
        ax.set_xticklabels(phase_labels, fontsize=7, rotation=15)
        ax.set_title(feat_name, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc(y_true: np.ndarray, y_scores: np.ndarray, signal_name: str) -> Figure:
    if len(np.unique(y_true)) < 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "Only one class — ROC not available",
                ha="center", va="center")
        ax.set_title(f"ROC — {signal_name}")
        return fig

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    best    = np.argmax(tpr - fpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2255aa", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Random classifier")
    ax.scatter(fpr[best], tpr[best], color="#cc0000", zorder=5, s=60,
               label=f"Best threshold = {thresholds[best]:.3f}")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve — {signal_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig