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

def _shade_phases(ax: plt.Axes, intervals: dict) -> list[mpatches.Patch]:
    """Shade phase intervals and return legend patches for phases that are present."""
    patches = []
    for phase_key, spans in intervals.items():
        if not spans:
            continue
        color, label = PHASE_COLORS.get(phase_key, ("#eeeeee", phase_key))
        for start, end in spans:
            ax.axvspan(start, end, color=color, alpha=0.22, zorder=0)
        patches.append(mpatches.Patch(color=color, alpha=0.5, label=label))
    return patches


def _stress_markers(ax: plt.Axes, t_centers: np.ndarray,
                    predictions: np.ndarray, y_pos: float) -> bool:
    """Draw stress markers. Returns True if any markers were drawn."""
    stressed = t_centers[predictions == 1]
    if len(stressed):
        ax.scatter(stressed, np.full(len(stressed), y_pos),
                   color="#cc0000", marker="v", s=28, zorder=6,
                   label="Predicted stress ▼")
        return True
    return False


# ── Stress peak detection ─────────────────────────────────────────────────────

def _find_stress_peaks(
    t_centers: np.ndarray,
    mean_scores: np.ndarray,
    threshold: float = 0.5,
    min_gap_s: float = 30.0,
) -> list[dict]:
    """
    Find stress episodes where mean_scores >= threshold.
    Returns list of dicts with keys: t_start, t_end, t_peak, score_peak.
    min_gap_s: merge episodes closer than this many seconds.
    """
    above = mean_scores >= threshold
    episodes = []
    in_ep    = False
    t_start  = 0.0

    for i, flag in enumerate(above):
        if flag and not in_ep:
            t_start = t_centers[i]
            in_ep   = True
        elif not flag and in_ep:
            t_end = t_centers[i - 1]
            seg   = mean_scores[
                (t_centers >= t_start) & (t_centers <= t_end)
            ]
            t_seg = t_centers[
                (t_centers >= t_start) & (t_centers <= t_end)
            ]
            pk_idx = int(np.argmax(seg))
            episodes.append({
                "t_start":    float(t_start),
                "t_end":      float(t_end),
                "t_peak":     float(t_seg[pk_idx]),
                "score_peak": float(seg[pk_idx]),
            })
            in_ep = False

    if in_ep:
        t_end = t_centers[-1]
        seg   = mean_scores[t_centers >= t_start]
        t_seg = t_centers[t_centers >= t_start]
        pk_idx = int(np.argmax(seg))
        episodes.append({
            "t_start":    float(t_start),
            "t_end":      float(t_end),
            "t_peak":     float(t_seg[pk_idx]),
            "score_peak": float(seg[pk_idx]),
        })

    # Merge episodes that are too close together
    merged = []
    for ep in episodes:
        if merged and ep["t_start"] - merged[-1]["t_end"] < min_gap_s:
            prev = merged[-1]
            if ep["score_peak"] > prev["score_peak"]:
                prev["t_peak"]     = ep["t_peak"]
                prev["score_peak"] = ep["score_peak"]
            prev["t_end"] = ep["t_end"]
        else:
            merged.append(dict(ep))
    return merged


def _draw_stress_peaks(
    ax: plt.Axes,
    t_centers: np.ndarray,
    mean_scores: np.ndarray,
    y_max: float,
    threshold: float = 0.5,
) -> bool:
    """
    Draw stress peak arrows and episode spans on ax.
    Returns True if any peaks were found.
    """
    episodes = _find_stress_peaks(t_centers, mean_scores, threshold)
    if not episodes:
        return False

    arrow_y = y_max * 1.02

    for ep in episodes:
        # Shade the stress episode span
        ax.axvspan(ep["t_start"], ep["t_end"],
                   color="#ff4444", alpha=0.18, zorder=1,
                   label="_nolegend_")
        # Vertical dashed lines at episode edges
        ax.axvline(ep["t_start"], color="#cc0000", lw=0.8,
                   ls=":", alpha=0.7, zorder=2)
        ax.axvline(ep["t_end"],   color="#cc0000", lw=0.8,
                   ls=":", alpha=0.7, zorder=2)
        # Arrow at peak
        ax.annotate(
            f"{ep['score_peak']:.2f}",
            xy=(ep["t_peak"], y_max),
            xytext=(ep["t_peak"], arrow_y),
            fontsize=7, color="#cc0000", ha="center",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#cc0000",
                lw=1.2,
            ),
            annotation_clip=False,
        )
    return True


# ── BPM panel ─────────────────────────────────────────────────────────────────

def _draw_bpm_panel(ax: plt.Axes, t_bpm: np.ndarray,
                    bpm: np.ndarray, phase_intervals: dict) -> list:
    ax.step(t_bpm, bpm, where="post", color="#e06000", lw=1.5, zorder=3,
            label="BPM")
    ax.fill_between(t_bpm, bpm, alpha=0.10, color="#e06000", step="post")

    bpm_mean = np.mean(bpm)
    thresh   = bpm_mean + np.std(bpm)

    ax.axhline(bpm_mean, color="#777777", lw=0.9, ls="--",
               label=f"Mean {bpm_mean:.0f} bpm")
    ax.axhline(thresh,   color="#cc0000", lw=0.9, ls=":",
               label=f"Stress thr. {thresh:.0f} bpm")

    for i in range(len(t_bpm) - 1):
        if bpm[i] > thresh:
            ax.axvspan(t_bpm[i], t_bpm[i + 1],
                       color="#ffcccc", alpha=0.45, zorder=0)
    if len(t_bpm) > 1 and bpm[-1] > thresh:
        ax.axvspan(t_bpm[-1], t_bpm[-1] + (t_bpm[-1] - t_bpm[-2]),
                   color="#ffcccc", alpha=0.45, zorder=0)

    diffs = np.diff(bpm)
    has_arrows = False
    for ji in np.where(np.abs(diffs) > 3)[0]:
        if bpm[ji] > thresh:
            ax.annotate("",
                        xy=(t_bpm[ji], bpm[ji]),
                        xytext=(t_bpm[ji], bpm[ji] + 2),
                        arrowprops=dict(arrowstyle="->",
                                        color="#cc0000", lw=1.2),
                        annotation_clip=True)
            has_arrows = True

    phase_patches = _shade_phases(ax, phase_intervals)
    ax.set_ylabel("BPM", fontsize=8)
    ax.set_title("Heart Rate (BPM)", fontsize=8, loc="left", pad=2)
    ax.grid(True, alpha=0.22)
    return phase_patches


# ── RR interval panel ─────────────────────────────────────────────────────────

def _draw_rr_panel(ax: plt.Axes, rr_times: np.ndarray,
                   rr_ms: np.ndarray, phase_intervals: dict) -> list:
    ax.plot(rr_times, rr_ms, color="#9933aa", lw=0.8,
            zorder=3, label="RR interval")

    rr_mean    = np.mean(rr_ms)
    low_thresh = rr_mean - np.std(rr_ms)

    ax.axhline(rr_mean,    color="#777777", lw=0.9, ls="--",
               label=f"Mean {rr_mean:.0f} ms")
    ax.axhline(low_thresh, color="#cc0000", lw=0.9, ls=":",
               label=f"Stress thr. {low_thresh:.0f} ms")

    mask = rr_ms < low_thresh
    has_short_rr = mask.any()
    if has_short_rr:
        ax.scatter(rr_times[mask], rr_ms[mask],
                   color="#cc0000", s=10, zorder=5,
                   label="Short RR (stress)")

    drops     = np.where(np.diff(rr_ms) < -30)[0]
    has_drops = len(drops) > 0
    for di in drops:
        ax.annotate("",
                    xy=(rr_times[di + 1], rr_ms[di + 1]),
                    xytext=(rr_times[di + 1], rr_ms[di + 1] + 25),
                    arrowprops=dict(arrowstyle="->",
                                    color="#cc0000", lw=1.0),
                    annotation_clip=True)

    phase_patches = _shade_phases(ax, phase_intervals)
    ax.set_ylabel("RR (ms)", fontsize=8)
    ax.set_title("RR Interval", fontsize=8, loc="left", pad=2)
    ax.grid(True, alpha=0.22)
    ax.invert_yaxis()
    return phase_patches


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

    panels = [(name, 2.2 if name in ("ECG", "BVP") else 1.0) for name in sig_names]
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

    # Compute mean stress score across all signals for peak detection
    mean_scores_global: np.ndarray | None = None
    t_global:           np.ndarray | None = None
    if stress_map:
        all_t      = max((v[0] for v in stress_map.values()), key=len)
        interp_all = []
        for t_c, preds in stress_map.values():
            # Use preds as binary scores if no probability available
            interp_all.append(
                np.interp(all_t, t_c, preds.astype(float),
                          left=0.0, right=0.0)
            )
        mean_scores_global = np.mean(np.array(interp_all), axis=0)
        t_global           = all_t

    # Track which legend items actually appear
    used_stress_marker = False
    used_stress_peak   = False
    used_bpm           = False
    used_rr            = False
    all_phase_patches: dict[str, mpatches.Patch] = {}

    for ax, (label, _) in zip(axs, panels):
        if label == "__BPM__":
            _draw_bpm_panel(ax, bpm_data[0], bpm_data[1], phase_intervals)
            ax.get_legend().remove()
            used_bpm = True
            # Collect phase patches
            for phase_key, spans in phase_intervals.items():
                if not spans:
                    continue
                color, lbl = PHASE_COLORS.get(phase_key, ("#eeeeee", phase_key))
                all_phase_patches[lbl] = mpatches.Patch(
                    color=color, alpha=0.5, label=lbl
                )
            continue
        if label == "__RR__":
            _draw_rr_panel(ax, rr_data[0], rr_data[1], phase_intervals)
            ax.get_legend().remove()
            used_rr = True
            for phase_key, spans in phase_intervals.items():
                if not spans:
                    continue
                color, lbl = PHASE_COLORS.get(phase_key, ("#eeeeee", phase_key))
                all_phase_patches[lbl] = mpatches.Patch(
                    color=color, alpha=0.5, label=lbl
                )
            continue

        sig  = signals[label]
        fs   = fs_map.get(label, 2000)
        data = sig.ravel()

        # Downsample for display (max 10 000 points)
        step = max(1, len(data) // 10_000)
        t    = np.arange(len(data))[::step] / fs
        ax.plot(t, data[::step], lw=0.55, color="#2255aa", rasterized=True)

        # Shade phases and collect patches
        for phase_key, spans in phase_intervals.items():
            if not spans:
                continue
            color, lbl = PHASE_COLORS.get(phase_key, ("#eeeeee", phase_key))
            for start, end in spans:
                ax.axvspan(start, end, color=color, alpha=0.22, zorder=0)
            all_phase_patches[lbl] = mpatches.Patch(
                color=color, alpha=0.5, label=lbl
            )

        y99 = float(np.nanpercentile(data, 99))

        if label in stress_map:
            t_c, preds = stress_map[label]
            had = _stress_markers(ax, t_c, preds, y99)
            if had:
                used_stress_marker = True

        if mean_scores_global is not None and t_global is not None:
            had_peaks = _draw_stress_peaks(
                ax, t_global, mean_scores_global, y99
            )
            if had_peaks:
                used_stress_peak = True

        # No per-panel legend
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9, loc="left", pad=2)
        ax.grid(True, alpha=0.22)

    axs[-1].set_xlabel("Time (s)", fontsize=9)

    # ── Single global legend ──────────────────────────────────────────────────
    legend_items: list = []

    # Signal line (generic — all signals use the same colour)
    legend_items.append(
        plt.Line2D([0], [0], color="#2255aa", lw=0.9, label="Signal")
    )

    # Phase patches (only those that actually appear)
    legend_items += list(all_phase_patches.values())

    # Stress marker
    if used_stress_marker:
        legend_items.append(
            plt.Line2D([0], [0], marker="v", color="w",
                       markerfacecolor="#cc0000", markersize=7,
                       label="Predicted stress ▼")
        )

    # Stress episode shading
    if used_stress_peak:
        legend_items.append(
            mpatches.Patch(color="#ff4444", alpha=0.4,
                           label="Stress episode")
        )

    # BPM line
    if used_bpm:
        legend_items.append(
            plt.Line2D([0], [0], color="#e06000", lw=1.5, label="BPM")
        )

    # RR line
    if used_rr:
        legend_items.append(
            plt.Line2D([0], [0], color="#9933aa", lw=0.8,
                       label="RR interval")
        )
        legend_items.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#cc0000", markersize=5,
                       label="Short RR (stress)")
        )

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=min(len(legend_items), 6),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.85,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.998])
    plt.subplots_adjust(hspace=0.5)
    return fig


# ── Boxplots ──────────────────────────────────────────────────────────────────

def plot_boxplots(phase_features: dict[str, np.ndarray],
                  signal_name: str) -> Figure:
    phase_labels = list(phase_features.keys())
    colors       = ["#5599dd", "#55bb77", "#8877dd", "#dd5555"]

    fig, axs = plt.subplots(1, FEATURE_DIM, figsize=(4 * FEATURE_DIM, 4))
    fig.suptitle(f"Feature Distribution — {signal_name}",
                 fontsize=12, fontweight="bold")

    for fi, feat_name in enumerate(FEATURE_NAMES):
        ax = axs[fi]
        data_per_phase = [
            phase_features[ph][:, fi]
            if phase_features[ph].ndim == 2
               and phase_features[ph].shape[1] > fi
            else np.array([])
            for ph in phase_labels
        ]
        bp = ax.boxplot(data_per_phase, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, col in zip(bp["boxes"], colors[:len(phase_labels)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        all_vals = np.concatenate(
            [d for d in data_per_phase if len(d) > 0]
        )
        if len(all_vals) > 0:
            ax.set_ylim(np.percentile(all_vals, 2),
                        np.percentile(all_vals, 98))

        ax.set_xticks(range(1, len(phase_labels) + 1))
        ax.set_xticklabels(phase_labels, fontsize=7, rotation=15)
        ax.set_title(feat_name, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc(y_true: np.ndarray, y_scores: np.ndarray,
             signal_name: str) -> Figure:
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
    ax.plot(fpr, tpr, color="#2255aa", lw=2,
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--",
            label="Random classifier")
    ax.scatter(fpr[best], tpr[best], color="#cc0000", zorder=5, s=60,
               label=f"Best thr. = {thresholds[best]:.3f}")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve — {signal_name}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig