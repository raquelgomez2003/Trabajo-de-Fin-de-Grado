"""
analysis_window.py
Analysis tab — stress prediction with configurable plot selection.
Supports Random Forest, LightGBM, SVM and KNN classifiers.
"""

from __future__ import annotations
import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.max_open_warning"] = 50  # suppress until 50 figures
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap

from app.core.models import (
    StressModel,
    extract_features_windowed,
    build_labels_from_intervals,
    build_physiological_labels,
    compute_rr,
    FEATURE_NAMES,
    FEATURE_DIM,
)
from app.core.config import RF_WINDOW_SEC, RF_STEP_SEC
from app.core.plotter import plot_boxplots, plot_roc


# ── Stress colormap (blue → yellow → red) ─────────────────────────────────────

STRESS_CMAP = LinearSegmentedColormap.from_list(
    "stress_cmap",
    ["#1a4fa0", "#4da6ff", "#ffff00", "#ff8800", "#cc0000"],
    N=256,
)

_FIG_W     = 12.0
_FIG_H     = 3.5
_LEFT      = 0.07
_RIGHT     = 0.87
_CBAR_LEFT = 0.88
_CBAR_W    = 0.02

# ── Available classifiers ─────────────────────────────────────────────────────

CLASSIFIERS: dict[str, type] = {}

try:
    from sklearn.ensemble import RandomForestClassifier
    CLASSIFIERS["Random Forest"] = RandomForestClassifier
except Exception:
    pass

try:
    import lightgbm as _lgb
    CLASSIFIERS["LightGBM"] = _lgb.LGBMClassifier
except Exception:
    try:
        from lightgbm import LGBMClassifier
        CLASSIFIERS["LightGBM"] = LGBMClassifier
    except Exception:
        pass

try:
    from sklearn.svm import SVC
    CLASSIFIERS["SVM"] = SVC
except Exception:
    pass

try:
    from sklearn.neighbors import KNeighborsClassifier
    CLASSIFIERS["KNN"] = KNeighborsClassifier
except Exception:
    pass


def _build_classifier(name: str):
    """Return a configured sklearn-compatible classifier by name."""
    if name == "Random Forest":
        return CLASSIFIERS[name](n_estimators=100, random_state=42, n_jobs=-1)
    if name == "LightGBM":
        return CLASSIFIERS[name](n_estimators=100, random_state=42,
                                  verbose=-1, n_jobs=-1)
    if name == "SVM":
        return CLASSIFIERS[name](kernel="rbf", probability=True,
                                  random_state=42, C=1.0)
    if name == "KNN":
        return CLASSIFIERS[name](n_neighbors=5, n_jobs=-1)
    raise ValueError(f"Unknown classifier: {name}")


def _fit_predict(clf, X_train, y_train, X_all):
    """Fit classifier and return (preds, scores).
    Imputes NaN values and uses plain numpy arrays."""
    import warnings
    from sklearn.impute import SimpleImputer

    X_train = np.asarray(X_train, dtype=float)
    X_all   = np.asarray(X_all,   dtype=float)

    # Impute NaN with column median (SVM/KNN cannot handle NaN)
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_all   = imp.transform(X_all)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_all)
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_all)[:, 1]
        else:
            scores = preds.astype(float)
    return preds, scores


# ── Helper: split features by phase ──────────────────────────────────────────

def _split_by_phase(
    X: np.ndarray,
    t_centers: np.ndarray,
    calming_intervals: list[tuple[float, float]],
    stress_intervals: list[tuple[float, float]],
) -> dict[str, np.ndarray]:
    mask_calming = np.zeros(len(t_centers), dtype=bool)
    mask_stress  = np.zeros(len(t_centers), dtype=bool)

    for s, e in calming_intervals:
        mask_calming |= (t_centers >= s) & (t_centers <= e)
    for s, e in stress_intervals:
        mask_stress  |= (t_centers >= s) & (t_centers <= e)

    mask_rest = ~mask_calming & ~mask_stress

    result = {}
    if mask_calming.any():
        result["calming"] = X[mask_calming]
    if mask_stress.any():
        result["stress"]  = X[mask_stress]
    if mask_rest.any():
        result["rest"]    = X[mask_rest]
    return result


# ── BPM from ECG ─────────────────────────────────────────────────────────────

def _compute_bpm_from_ecg(
    ecg: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import find_peaks, butter, filtfilt

    ecg = np.asarray(ecg, dtype=float).ravel()
    nyq  = fs / 2.0
    low  = 0.5 / nyq
    high = min(40.0 / nyq, 0.99)
    b, a = butter(2, [low, high], btype="band")
    ecg_f = filtfilt(b, a, ecg)

    peak_val = np.max(np.abs(ecg_f))
    if peak_val == 0:
        return np.array([]), np.array([])
    ecg_n = ecg_f / peak_val

    min_dist   = int(0.4 * fs)
    height_thr = 0.3
    peaks, _   = find_peaks(ecg_n, height=height_thr, distance=min_dist)
    if len(peaks) < 2:
        return np.array([]), np.array([])

    rr_samples = np.diff(peaks)
    rr_sec     = rr_samples / fs
    bpm        = 60.0 / rr_sec
    t_bpm      = (peaks[:-1] + peaks[1:]) / 2.0 / fs

    valid = (bpm >= 20) & (bpm <= 250)
    return t_bpm[valid], bpm[valid]


def _plot_ecg_bpm(
    ecg: np.ndarray,
    fs: float,
    stress_intervals: list[tuple[float, float]],
    calming_intervals: list[tuple[float, float]],
) -> plt.Figure:
    t_bpm, bpm = _compute_bpm_from_ecg(ecg, fs)
    fig, ax = plt.subplots(figsize=(11, 3.2))

    if len(t_bpm) == 0:
        ax.text(0.5, 0.5,
                "No R-peaks detected.\nCheck ECG signal.",
                ha="center", va="center", fontsize=10, color="#cc4400")
        ax.set_title("Heart Rate (BPM) from ECG", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig

    k          = min(11, len(bpm))
    kernel     = np.ones(k) / k
    bpm_smooth = np.convolve(bpm, kernel, mode="same")
    bpm_mean   = float(np.mean(bpm))
    bpm_std    = float(np.std(bpm))
    thr_high   = bpm_mean + bpm_std
    thr_low    = bpm_mean - bpm_std

    ax.scatter(t_bpm, bpm, s=6, color="#e06000", alpha=0.35, zorder=2,
               label="Instantaneous BPM")
    ax.plot(t_bpm, bpm_smooth, color="#b04000", lw=1.8, zorder=4,
            label=f"Smoothed BPM ({k}-beat avg)")
    ax.fill_between(t_bpm, bpm_mean, bpm_smooth,
                    where=bpm_smooth >= bpm_mean,
                    color="#ffaa44", alpha=0.20, zorder=1)
    ax.fill_between(t_bpm, bpm_mean, bpm_smooth,
                    where=bpm_smooth < bpm_mean,
                    color="#4499ff", alpha=0.15, zorder=1)
    ax.axhline(bpm_mean, color="#555555", lw=1.0, ls="--",
               label=f"Mean  {bpm_mean:.0f} bpm", zorder=3)
    ax.axhline(thr_high, color="#cc0000", lw=1.0, ls=":",
               label=f"+1σ  {thr_high:.0f} bpm", zorder=3)
    ax.axhline(thr_low,  color="#0055cc", lw=1.0, ls=":",
               label=f"−1σ  {thr_low:.0f} bpm",  zorder=3)

    in_stress, seg_start = False, 0.0
    for i in range(len(t_bpm)):
        above = bpm_smooth[i] > thr_high
        if above and not in_stress:
            seg_start, in_stress = t_bpm[i], True
        elif not above and in_stress:
            ax.axvspan(seg_start, t_bpm[i], color="#ffcccc", alpha=0.55, zorder=0)
            in_stress = False
    if in_stress:
        ax.axvspan(seg_start, t_bpm[-1], color="#ffcccc", alpha=0.55, zorder=0)

    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.22, zorder=0)
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.18, zorder=0)

    ax.set_ylim(max(20, bpm_mean - 4 * bpm_std),
                min(250, bpm_mean + 4 * bpm_std))
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("BPM", fontsize=9)
    ax.set_title("Heart Rate (BPM) from ECG — red zone: above stress threshold",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.75, ncol=2)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


# ── Respiratory rate from RESP ────────────────────────────────────────────────

def _compute_resp_rate(
    resp: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import find_peaks, butter, filtfilt

    resp = np.asarray(resp, dtype=float).ravel()
    nyq  = fs / 2.0
    low  = max(0.1 / nyq, 1e-4)
    high = min(0.8 / nyq, 0.99)
    b, a = butter(2, [low, high], btype="band")
    resp_f = filtfilt(b, a, resp)

    peak_val = np.max(np.abs(resp_f))
    if peak_val == 0:
        return np.array([]), np.array([])
    resp_n = resp_f / peak_val

    min_dist   = int(1.5 * fs)
    height_thr = 0.2
    peaks, _   = find_peaks(resp_n, height=height_thr, distance=min_dist)
    if len(peaks) < 2:
        return np.array([]), np.array([])

    intervals_sec = np.diff(peaks) / fs
    rpm           = 60.0 / intervals_sec
    t_rpm         = (peaks[:-1] + peaks[1:]) / 2.0 / fs

    valid = (rpm >= 1) & (rpm <= 60)
    return t_rpm[valid], rpm[valid]


def _plot_resp_rate(
    resp: np.ndarray,
    fs: float,
    stress_intervals: list[tuple[float, float]],
    calming_intervals: list[tuple[float, float]],
) -> plt.Figure:
    t_rpm, rpm = _compute_resp_rate(resp, fs)
    fig, ax = plt.subplots(figsize=(11, 3.2))

    if len(t_rpm) == 0:
        ax.text(0.5, 0.5,
                "No respiratory cycles detected.\nCheck RESP signal.",
                ha="center", va="center", fontsize=10, color="#cc4400")
        ax.set_title("Respiratory Rate (rpm) from RESP",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig

    k          = min(7, len(rpm))
    kernel     = np.ones(k) / k
    rpm_smooth = np.convolve(rpm, kernel, mode="same")
    rpm_mean   = float(np.mean(rpm))
    rpm_std    = float(np.std(rpm))
    thr_high   = rpm_mean + rpm_std
    thr_low    = rpm_mean - rpm_std

    ax.scatter(t_rpm, rpm, s=6, color="#2288cc", alpha=0.35, zorder=2,
               label="Instantaneous rpm")
    ax.plot(t_rpm, rpm_smooth, color="#115588", lw=1.8, zorder=4,
            label=f"Smoothed rpm ({k}-cycle avg)")
    ax.fill_between(t_rpm, rpm_mean, rpm_smooth,
                    where=rpm_smooth >= rpm_mean,
                    color="#ffaa44", alpha=0.20, zorder=1)
    ax.fill_between(t_rpm, rpm_mean, rpm_smooth,
                    where=rpm_smooth < rpm_mean,
                    color="#88ccff", alpha=0.15, zorder=1)
    ax.axhline(rpm_mean, color="#555555", lw=1.0, ls="--",
               label=f"Mean  {rpm_mean:.1f} rpm", zorder=3)
    ax.axhline(thr_high, color="#cc0000", lw=1.0, ls=":",
               label=f"+1σ  {thr_high:.1f} rpm", zorder=3)
    ax.axhline(thr_low,  color="#0055cc", lw=1.0, ls=":",
               label=f"−1σ  {thr_low:.1f} rpm",  zorder=3)
    ax.axhspan(12, 20, color="#ddffdd", alpha=0.25, zorder=0,
               label="Normal range (12–20 rpm)")

    in_stress, seg_start = False, 0.0
    for i in range(len(t_rpm)):
        above = rpm_smooth[i] > thr_high
        if above and not in_stress:
            seg_start, in_stress = t_rpm[i], True
        elif not above and in_stress:
            ax.axvspan(seg_start, t_rpm[i], color="#ffcccc", alpha=0.55, zorder=0)
            in_stress = False
    if in_stress:
        ax.axvspan(seg_start, t_rpm[-1], color="#ffcccc", alpha=0.55, zorder=0)

    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.22, zorder=0)
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.18, zorder=0)

    ax.set_ylim(max(0, rpm_mean - 4 * rpm_std),
                min(60, rpm_mean + 4 * rpm_std))
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("rpm", fontsize=9)
    ax.set_title("Respiratory Rate (rpm) from RESP — red zone: above stress threshold",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.75, ncol=2)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


# ── Global tab plots ──────────────────────────────────────────────────────────

def _plot_global_stress_probability(
    stress_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stress_intervals: list[tuple[float, float]],
    calming_intervals: list[tuple[float, float]],
    model_name: str = "",
) -> plt.Figure:
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    ax  = fig.add_axes([_LEFT, 0.18, _RIGHT - _LEFT, 0.72])

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(stress_map), 1)))
    all_scores: list[tuple[np.ndarray, np.ndarray]] = []

    for (sig_name, (t_centers, preds, scores)), color in zip(
        stress_map.items(), colors
    ):
        ax.plot(t_centers, scores, lw=0.9, alpha=0.55,
                color=color, label=sig_name)
        all_scores.append((t_centers, scores))

    if len(all_scores) > 1:
        t_ref  = max(all_scores, key=lambda x: len(x[0]))[0]
        interp = np.array([
            np.interp(t_ref, t, s, left=np.nan, right=np.nan)
            for t, s in all_scores
        ])
        mean_score = np.nanmean(interp, axis=0)
        ax.plot(t_ref, mean_score, lw=2.2, color="black",
                label="Mean", zorder=5)
        ax.fill_between(t_ref, mean_score, alpha=0.15, color="black")

    ax.axhline(0.5, color="#cc0000", lw=0.9, ls="--",
               label="Threshold 0.5", zorder=4)

    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.30, zorder=0)
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.30, zorder=0)

    title = "Combined stress probability — all signals"
    if model_name:
        title += f"  [{model_name}]"
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("P(stress)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0, framealpha=0.75)
    ax.grid(True, alpha=0.25)
    return fig


def _plot_global_heatmap_mean(
    stress_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stress_intervals: list[tuple[float, float]],
    calming_intervals: list[tuple[float, float]],
    model_name: str = "",
) -> plt.Figure:
    sig_names = list(stress_map.keys())
    if not sig_names:
        fig = plt.figure(figsize=(_FIG_W, _FIG_H))
        ax  = fig.add_axes([_LEFT, 0.18, _RIGHT - _LEFT, 0.72])
        ax.text(0.5, 0.5, "No results to display.", ha="center", va="center")
        return fig

    t_ref        = max((stress_map[s][0] for s in sig_names), key=len)
    t_min, t_max = t_ref[0], t_ref[-1]

    rows = []
    for sig in sig_names:
        t_sig, _, scores = stress_map[sig]
        if len(t_sig) > 1:
            rows.append(np.interp(t_ref, t_sig, scores,
                                  left=np.nan, right=np.nan))
    mean_row = np.nanmean(np.array(rows), axis=0)

    fig      = plt.figure(figsize=(_FIG_W, _FIG_H))
    ax_heat  = fig.add_axes([_LEFT, 0.32, _RIGHT - _LEFT, 0.52])
    ax_phase = fig.add_axes([_LEFT, 0.10, _RIGHT - _LEFT, 0.18])
    ax_cbar  = fig.add_axes([_CBAR_LEFT, 0.32, _CBAR_W, 0.52])

    im = ax_heat.imshow(
        mean_row[np.newaxis, :],
        aspect="auto", cmap=STRESS_CMAP, vmin=0.0, vmax=1.0,
        extent=[t_min, t_max, 0.5, -0.5], interpolation="bilinear",
    )
    ax_heat.set_yticks([0])
    ax_heat.set_yticklabels(["Mean"], fontsize=9)
    ax_heat.set_xticks([])
    title = "Mean stress probability heatmap"
    if model_name:
        title += f"  [{model_name}]"
    ax_heat.set_title(title, fontsize=11, fontweight="bold", pad=5)

    for s, e in calming_intervals:
        ax_heat.axvline(s, color="#00aa44", lw=1.0, ls="--", alpha=0.7)
        ax_heat.axvline(e, color="#00aa44", lw=1.0, ls="--", alpha=0.7)
    for s, e in stress_intervals:
        ax_heat.axvline(s, color="#cc0000", lw=1.0, ls="--", alpha=0.7)
        ax_heat.axvline(e, color="#cc0000", lw=1.0, ls="--", alpha=0.7)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("P(stress)", fontsize=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)

    ax_phase.set_xlim(t_min, t_max)
    ax_phase.set_ylim(0, 1)
    ax_phase.set_yticks([])
    ax_phase.set_xlabel("Time (s)", fontsize=9)
    ax_phase.axhspan(0, 1, color="#dddddd", alpha=0.4)
    for s, e in calming_intervals:
        ax_phase.axvspan(s, e, color="#b6f0c8", alpha=0.9)
        ax_phase.text((s + e) / 2, 0.5, "Calming",
                      ha="center", va="center", fontsize=7,
                      color="#1a6b3a", fontweight="bold")
    for s, e in stress_intervals:
        ax_phase.axvspan(s, e, color="#ffb3b3", alpha=0.9)
        ax_phase.text((s + e) / 2, 0.5, "Stress",
                      ha="center", va="center", fontsize=7,
                      color="#aa2222", fontweight="bold")
    for spine in ["top", "right", "left"]:
        ax_phase.spines[spine].set_visible(False)
    return fig


# ── Per-signal plot functions ─────────────────────────────────────────────────

def _plot_stress_timeline(
    t_centers: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    sig_name: str,
    stress_intervals: list[tuple[float, float]],
    model_name: str = "",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(t_centers, scores, alpha=0.35, color="#2255aa",
                    label="Stress probability")
    ax.plot(t_centers, scores, lw=0.8, color="#2255aa")
    ax.axhline(0.5, color="#cc0000", lw=0.8, ls="--",
               label="Decision threshold 0.5")

    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.4,
                   label="Session stress interval")

    stressed_t = t_centers[preds == 1]
    if len(stressed_t):
        ax.scatter(stressed_t, np.full(len(stressed_t), 0.02),
                   color="#cc0000", marker="|", s=40, zorder=5,
                   label="Predicted stress")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("P(stress)", fontsize=9)
    title = f"Stress probability — {sig_name}"
    if model_name:
        title += f"  [{model_name}]"
    ax.set_title(title, fontsize=10, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ── Popup helper ──────────────────────────────────────────────────────────────

def _open_popup(fig: plt.Figure, title: str) -> None:
    popup = ctk.CTkToplevel()
    popup.title(title)
    popup.geometry("1100x520")
    popup.grab_set()
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    toolbar_frame = ctk.CTkFrame(popup, fg_color="transparent")
    toolbar_frame.pack(side="bottom", fill="x")
    NavigationToolbar2Tk(canvas, toolbar_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
    popup.protocol("WM_DELETE_WINDOW", popup.destroy)


# ── Main analysis window ──────────────────────────────────────────────────────

class AnalysisWindow(ctk.CTkFrame):

    PLOT_TYPES = [
        ("Boxplots",             "boxplots"),
        ("ROC curve",            "roc"),
        ("Stress timeline",      "timeline"),
        ("BPM from ECG",         "bpm_ecg"),
        ("Resp. rate from RESP", "resp_rate"),
    ]

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._figs: list[plt.Figure] = []

        self._signals:           dict[str, np.ndarray]     = {}
        self._fs_map:            dict[str, float]          = {}
        self._stress_intervals:  list[tuple[float, float]] = []
        self._calming_intervals: list[tuple[float, float]] = []

        self._sig_vars:   dict[str, ctk.BooleanVar] = {}
        self._plot_vars:  dict[str, ctk.BooleanVar] = {}
        self._clf_vars:   dict[str, ctk.BooleanVar] = {}
        self._window_var: ctk.StringVar | None = None
        self._step_var:   ctk.StringVar | None = None
        self._status_var: ctk.StringVar | None = None

        self._results_frame: ctk.CTkFrame | None = None
        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._ctrl = ctk.CTkScrollableFrame(self, width=230)
        self._ctrl.grid(row=0, column=0, sticky="ns", padx=(6, 2), pady=6)
        ctk.CTkLabel(self._ctrl, text="Analysis controls",
                     font=("Arial", 13, "bold")).pack(anchor="w", pady=(8, 4))
        self._build_controls()

        self._results_frame = ctk.CTkFrame(self)
        self._results_frame.grid(row=0, column=1, sticky="nsew",
                                 padx=(2, 6), pady=6)
        self._results_frame.grid_columnconfigure(0, weight=1)
        self._results_frame.grid_rowconfigure(0, weight=1)
        ctk.CTkLabel(
            self._results_frame,
            text="Configure the controls on the left and press Run.",
            text_color="gray", font=("Arial", 13),
        ).pack(pady=40)

    def _build_controls(self):
        ctrl = self._ctrl

        # Signals
        self._sig_section = ctk.CTkFrame(ctrl, fg_color="transparent")
        self._sig_section.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(self._sig_section, text="Signals to analyse",
                     font=("Arial", 11, "bold")).pack(anchor="w")
        ctk.CTkLabel(self._sig_section, text="(load a subject first)",
                     text_color="gray", font=("Arial", 10)).pack(anchor="w")

        # Classifiers
        ctk.CTkLabel(ctrl, text="Classifiers",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 2))
        self._clf_vars = {}
        defaults = {"Random Forest": True, "LightGBM": False,
                    "SVM": False, "KNN": False}
        for clf_name in CLASSIFIERS:
            var = ctk.BooleanVar(value=defaults.get(clf_name, False))
            self._clf_vars[clf_name] = var
            ctk.CTkCheckBox(ctrl, text=clf_name, variable=var,
                            font=("Arial", 11)).pack(anchor="w", padx=4, pady=1)

        if not CLASSIFIERS:
            ctk.CTkLabel(ctrl, text="No classifiers available.\nInstall sklearn/lightgbm.",
                         text_color="#cc4400", font=("Arial", 10)).pack(anchor="w", padx=4)

        # Plot types
        ctk.CTkLabel(ctrl, text="Plot types",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 2))
        self._plot_vars = {}
        for label, key in self.PLOT_TYPES:
            var = ctk.BooleanVar(value=True)
            self._plot_vars[key] = var
            ctk.CTkCheckBox(ctrl, text=label, variable=var,
                            font=("Arial", 11)).pack(anchor="w", padx=4, pady=1)

        # Window / step
        ctk.CTkLabel(ctrl, text="Window size (s)",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 2))
        self._window_var = ctk.StringVar(value=str(RF_WINDOW_SEC))
        ctk.CTkEntry(ctrl, textvariable=self._window_var,
                     height=28).pack(fill="x", padx=4)

        ctk.CTkLabel(ctrl, text="Step size (s)",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(6, 2))
        self._step_var = ctk.StringVar(value=str(RF_STEP_SEC))
        ctk.CTkEntry(ctrl, textvariable=self._step_var,
                     height=28).pack(fill="x", padx=4)

        ctk.CTkButton(ctrl, text="▶  Run analysis",
                      height=38, fg_color="#336699", hover_color="#224477",
                      command=self._run).pack(fill="x", padx=4, pady=(14, 4))

        self._status_var = ctk.StringVar(value="")
        ctk.CTkLabel(ctrl, textvariable=self._status_var,
                     text_color="gray", font=("Arial", 10),
                     wraplength=210).pack(padx=4, pady=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_data(
        self,
        signals: dict[str, np.ndarray],
        fs_map: dict[str, float],
        stress_intervals: list[tuple[float, float]],
        calming_intervals: list[tuple[float, float]],
    ) -> None:
        self._signals           = signals
        self._fs_map            = fs_map
        self._stress_intervals  = stress_intervals
        self._calming_intervals = calming_intervals
        self._refresh_signal_checkboxes()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _refresh_signal_checkboxes(self):
        for w in self._sig_section.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._sig_section, text="Signals to analyse",
                     font=("Arial", 11, "bold")).pack(anchor="w")
        self._sig_vars = {}
        for sig in self._signals:
            var = ctk.BooleanVar(value=True)
            self._sig_vars[sig] = var
            ctk.CTkCheckBox(self._sig_section, text=sig, variable=var,
                            font=("Arial", 11)).pack(anchor="w", padx=4, pady=1)
        if self._status_var:
            self._status_var.set(f"{len(self._signals)} signal(s) ready.")

    def _selected_signals(self) -> list[str]:
        return [s for s, v in self._sig_vars.items() if v.get()]

    def _selected_plots(self) -> set[str]:
        return {k for k, v in self._plot_vars.items() if v.get()}

    def _selected_classifiers(self) -> list[str]:
        return [n for n, v in self._clf_vars.items() if v.get()]

    def _get_window_step(self) -> tuple[float, float]:
        try:
            win  = float(self._window_var.get())
            step = float(self._step_var.get())
            assert win > 0 and step > 0 and step <= win
            return win, step
        except Exception:
            return RF_WINDOW_SEC, RF_STEP_SEC

    def _set_status(self, msg: str) -> None:
        if self._status_var:
            self._status_var.set(msg)
        self.update_idletasks()

    def _clear_results(self):
        """Clear only the results area. Does NOT wipe loaded signal data."""
        # Close all figures first
        for fig in self._figs:
            try:
                plt.close(fig)
            except Exception:
                pass
        self._figs.clear()
        plt.close("all")

        # Collect children to destroy
        children = list(self._results_frame.winfo_children())

        def _do_destroy():
            for w in children:
                try:
                    if w.winfo_exists():
                        w.destroy()
                except Exception:
                    pass

        # Defer destruction to after current event is processed
        self._results_frame.after_idle(_do_destroy)

        self._ctrl.grid()
        self.grid_columnconfigure(0, weight=0)

    def reset_for_new_subject(self):
        """Called from app_window on reset. Clears results AND signal data."""
        self._clear_results()
        self._signals           = {}
        self._fs_map            = {}
        self._stress_intervals  = []
        self._calming_intervals = []
        self._sig_vars          = {}
        self._refresh_signal_checkboxes()

    def _add_section_header_in(self, parent, sig_name, preds, scores, row,
                                model_name=""):
        n_stressed = int(preds.sum())
        n_total    = len(preds)
        label_text = (f"  [{model_name}]  " if model_name else "  ")
        label_text += (f"Stressed windows: {n_stressed} / {n_total}   ·   "
                       f"Mean stress prob: {scores.mean():.2f}")
        frame = ctk.CTkFrame(parent, fg_color="#336699", corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=6, pady=(8, 2))
        ctk.CTkLabel(frame, text=label_text,
                     font=("Arial", 11, "bold"),
                     text_color="white").pack(side="left", padx=8, pady=6)

    def _add_plot_in(self, parent, title, fig, row):
        self._figs.append(fig)
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        outer.grid(row=row, column=0, sticky="ew", padx=6, pady=4)
        outer.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(outer, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        if title:
            ctk.CTkLabel(header, text=title,
                         font=("Arial", 11, "bold"), anchor="w").grid(
                row=0, column=0, sticky="w", padx=4)

        popup_title = title or "Plot"
        ctk.CTkButton(header, text="⛶  Expand", width=90, height=24,
                      font=("Arial", 10), fg_color="#4526FA",
                      hover_color="#A3A3A3",
                      command=lambda f=fig, t=popup_title: _open_popup(f, t)
                      ).grid(row=0, column=1, sticky="e", padx=4)

        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="ew")

    def _add_section_label(self, parent, text, row):
        ctk.CTkLabel(parent, text=text,
                     font=("Arial", 12, "bold"), anchor="w").grid(
            row=row, column=0, sticky="ew", padx=8, pady=(10, 2))

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        self._clear_results()

        selected_sigs  = self._selected_signals()
        selected_plots = self._selected_plots()
        selected_clfs  = self._selected_classifiers()
        window_sec, step_sec = self._get_window_step()

        if not selected_sigs:
            self._set_status("Select at least one signal.")
            return {}
        if not selected_plots:
            self._set_status("Select at least one plot type.")
            return {}
        if not self._signals:
            self._set_status("Load a subject first.")
            return {}
        if not selected_clfs:
            self._set_status("Select at least one classifier.")
            return {}

        self._set_status("Running…")
        self.update_idletasks()

        signals = {s: self._signals[s] for s in selected_sigs
                   if s in self._signals}

        t_phys, y_phys = build_physiological_labels(
            signals, self._fs_map, window_sec, step_sec
        )

        # Clear any leftover placeholder widgets before building tab view
        for w in list(self._results_frame.winfo_children()):
            try:
                if w.winfo_exists():
                    w.destroy()
            except Exception:
                pass

        # Build tab view
        tab_view = ctk.CTkTabview(self._results_frame)
        tab_view.pack(fill="both", expand=True, padx=2, pady=2)

        # One tab per signal
        tabs: dict[str, ctk.CTkScrollableFrame] = {}
        for sig_name in selected_sigs:
            if sig_name in signals:
                tab_view.add(sig_name)
                scroll = ctk.CTkScrollableFrame(tab_view.tab(sig_name))
                scroll.pack(fill="both", expand=True)
                scroll.grid_columnconfigure(0, weight=1)
                tabs[sig_name] = scroll

        # Global tab always last
        tab_view.add("Global")
        global_scroll = ctk.CTkScrollableFrame(tab_view.tab("Global"))
        global_scroll.pack(fill="both", expand=True)
        global_scroll.grid_columnconfigure(0, weight=1)

        # Row counters per tab
        tab_rows  = {sig: 0 for sig in tabs}
        g_row     = 0
        n_done    = 0

        # ── Run each selected classifier ──────────────────────────────────────
        for clf_name in selected_clfs:
            self._set_status(f"Running {clf_name}…")
            self.update_idletasks()

            # Per-signal loop
            stress_map: dict[str, tuple] = {}
            per_sig:    dict[str, dict]  = {}

            for sig_name in selected_sigs:
                if sig_name not in signals:
                    continue

                fs = self._fs_map.get(sig_name, 2000)
                X_all, t_centers = extract_features_windowed(
                    signals[sig_name], fs, sig_name, window_sec, step_sec
                )
                if len(X_all) == 0:
                    continue

                n_win = min(len(X_all), len(t_phys))
                if n_win == 0:
                    continue

                X_train = X_all[:n_win]
                y_train = y_phys[:n_win]
                if len(np.unique(y_train)) < 2:
                    continue

                try:
                    clf = _build_classifier(clf_name)
                    preds, scores = _fit_predict(clf, X_train, y_train, X_all)
                except Exception as ex:
                    self._set_status(f"{clf_name} failed: {ex}")
                    continue

                stress_map[sig_name] = (t_centers, preds, scores)
                per_sig[sig_name] = {
                    "X_all":       X_all,
                    "t_centers":   t_centers,
                    "preds":       preds,
                    "scores":      scores,
                    "y_session":   build_labels_from_intervals(
                                       t_centers, self._stress_intervals),
                    "phase_feats": _split_by_phase(
                                       X_all, t_centers,
                                       self._calming_intervals,
                                       self._stress_intervals),
                }

            if not stress_map:
                continue

            # ── Fill per-signal tabs for this classifier ──────────────────────
            for sig_name, res in per_sig.items():
                if sig_name not in tabs:
                    continue
                parent    = tabs[sig_name]
                t_centers = res["t_centers"]
                preds     = res["preds"]
                scores    = res["scores"]
                row       = tab_rows[sig_name]
                fs        = self._fs_map.get(sig_name, 2000)

                # Separator if multiple classifiers
                if len(selected_clfs) > 1:
                    self._add_section_label(
                        parent, f"── {clf_name} ──", row
                    )
                    row += 1

                self._add_section_header_in(
                    parent, sig_name, preds, scores, row, clf_name
                )
                row += 1

                if "boxplots" in selected_plots:
                    fig = plot_boxplots(res["phase_feats"], sig_name)
                    self._add_plot_in(
                        parent,
                        f"Feature distributions  [{clf_name}]", fig, row
                    )
                    row += 1

                if "roc" in selected_plots:
                    fig = plot_roc(res["y_session"], scores, sig_name)
                    self._add_plot_in(
                        parent, f"ROC curve  [{clf_name}]", fig, row
                    )
                    row += 1

                if "timeline" in selected_plots:
                    fig = _plot_stress_timeline(
                        t_centers, preds, scores, sig_name,
                        self._stress_intervals, clf_name,
                    )
                    self._add_plot_in(
                        parent, f"Stress timeline  [{clf_name}]", fig, row
                    )
                    row += 1

                if "bpm_ecg" in selected_plots and sig_name == "ECG":
                    # BPM is signal-based, show only once (first classifier)
                    if clf_name == selected_clfs[0]:
                        fig = _plot_ecg_bpm(
                            signals[sig_name], fs,
                            self._stress_intervals, self._calming_intervals,
                        )
                        self._add_plot_in(
                            parent, "Heart Rate (BPM) from ECG", fig, row
                        )
                        row += 1

                if "resp_rate" in selected_plots and sig_name == "RESP":
                    if clf_name == selected_clfs[0]:
                        fig = _plot_resp_rate(
                            signals[sig_name], fs,
                            self._stress_intervals, self._calming_intervals,
                        )
                        self._add_plot_in(
                            parent,
                            "Respiratory Rate (rpm) from RESP", fig, row
                        )
                        row += 1

                tab_rows[sig_name] = row

            # ── Global tab section for this classifier ────────────────────────
            if len(selected_clfs) > 1:
                self._add_section_label(
                    global_scroll, f"── {clf_name} ──", g_row
                )
                g_row += 1

            self._add_section_label(
                global_scroll,
                f"Combined stress probability  [{clf_name}]",
                g_row,
            )
            g_row += 1
            fig_prob = _plot_global_stress_probability(
                stress_map, self._stress_intervals,
                self._calming_intervals, clf_name,
            )
            self._add_plot_in(global_scroll, "", fig_prob, g_row)
            g_row += 1

            self._add_section_label(
                global_scroll,
                f"Mean stress heatmap  [{clf_name}]",
                g_row,
            )
            g_row += 1
            fig_heat = _plot_global_heatmap_mean(
                stress_map, self._stress_intervals,
                self._calming_intervals, clf_name,
            )
            self._add_plot_in(global_scroll, "", fig_heat, g_row)
            g_row += 1

            n_done += 1

        # BPM and RESP overviews in Global tab (once, independent of classifier)
        if "bpm_ecg" in selected_plots and "ECG" in signals:
            self._add_section_label(
                global_scroll, "Heart Rate (BPM) from ECG — global overview", g_row
            )
            g_row += 1
            fig_bpm = _plot_ecg_bpm(
                signals["ECG"], self._fs_map.get("ECG", 2000),
                self._stress_intervals, self._calming_intervals,
            )
            self._add_plot_in(global_scroll, "", fig_bpm, g_row)
            g_row += 1

        if "resp_rate" in selected_plots and "RESP" in signals:
            self._add_section_label(
                global_scroll,
                "Respiratory Rate (rpm) from RESP — global overview", g_row
            )
            g_row += 1
            fig_resp = _plot_resp_rate(
                signals["RESP"], self._fs_map.get("RESP", 2000),
                self._stress_intervals, self._calming_intervals,
            )
            self._add_plot_in(global_scroll, "", fig_resp, g_row)

        if n_done == 0:
            ctk.CTkLabel(
                self._results_frame,
                text="Not enough data or all classifiers failed.",
                text_color="#cc4400", font=("Arial", 12),
            ).pack(pady=20)
            self._set_status("No results.")
            return {}

        self._set_status(
            f"Done — {n_done} classifier(s), "
            f"{len(tabs)} signal(s) analysed."
        )
        self._ctrl.grid_remove()
        self.grid_columnconfigure(0, weight=0)
        return {}