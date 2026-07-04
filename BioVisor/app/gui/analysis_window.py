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
plt.rcParams["figure.max_open_warning"] = 50
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap
import os
import csv
from datetime import datetime
from tkinter import messagebox, filedialog

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
from app.core.plotter import plot_roc


# ── Stress colormap ───────────────────────────────────────────────────────────

STRESS_CMAP = LinearSegmentedColormap.from_list(
    "stress_cmap",
    ["#1a4fa0", "#4da6ff", "#ffff00", "#ff8800", "#cc0000"],
    N=256,
)

# ── Layout compartido para las graficas globales ──────────────────────────────
# La grafica de probabilidad, el heatmap y la barra de fases usan EXACTAMENTE
# el mismo margen izquierdo (_LEFT) y el mismo borde derecho (_RIGHT), de modo
# que quedan perfectamente alineados en el eje de tiempo. La colorbar se coloca
# justo a la derecha (_CBAR_LEFT), sin dejar margen blanco.
_FIG_W     = 12.0
_FIG_H     = 3.5
_LEFT      = 0.07
_RIGHT     = 0.90     # borde derecho del area de dibujo (prob + heatmap + fases)
_CBAR_LEFT = 0.915    # colorbar pegada a la derecha del heatmap
_CBAR_W    = 0.018

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
    if name == "Random Forest":
        return CLASSIFIERS[name](n_estimators=100, random_state=42, n_jobs=-1)
    if name == "LightGBM":
        return CLASSIFIERS[name](n_estimators=100, random_state=42,
                                  verbose=-1, n_jobs=-1)
    if name == "SVM":
        return CLASSIFIERS[name](kernel="rbf", probability=True,
                                  random_state=42, C=1.0, gamma="scale")
    if name == "KNN":
        return CLASSIFIERS[name](n_neighbors=5, n_jobs=-1)
    raise ValueError(f"Unknown classifier: {name}")


def _fit_predict(clf, X_train, y_train, X_all):
    import warnings
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, dtype=float)
    X_all   = np.asarray(X_all,   dtype=float)

    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_all   = imp.transform(X_all)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_all   = scaler.transform(X_all)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_all)
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_all)[:, 1]
        else:
            df = clf.decision_function(X_all)
            scores = (df - df.min()) / (df.max() - df.min() + 1e-9)
    return preds, scores


# ── CSV export ────────────────────────────────────────────────────────────────

def _export_stress_csv(all_stress_maps, parent_widget):
    """
    Export the full global stress probability time series per classifier.
    Each column is the mean score across all signals at each time step
    (the bold black line in the global plot).
    Rows = time steps, columns = time_s + one score column per classifier.
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Stress_Probability_{ts}.csv"
    path     = filedialog.asksaveasfilename(
        initialfile=filename, defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save stress probability CSV",
    )
    if not path:
        return
    if not all_stress_maps:
        messagebox.showwarning("Save", "No data to export.")
        return

    try:
        # Build a common time grid from the longest signal across all classifiers
        all_t = []
        for clf_data in all_stress_maps.values():
            for t_c, _, _ in clf_data.values():
                all_t.append(t_c)
        t_ref = max(all_t, key=len)

        # For each classifier compute mean score across all signals (bold line)
        clf_means = {}
        for clf_name, clf_data in all_stress_maps.items():
            rows = []
            for t_c, _, scores in clf_data.values():
                if len(t_c) > 1:
                    rows.append(
                        np.interp(t_ref, t_c, scores,
                                  left=np.nan, right=np.nan)
                    )
            if rows:
                clf_means[clf_name] = np.nanmean(np.array(rows), axis=0)

        if not clf_means:
            messagebox.showwarning("Save", "No data to export.")
            return

        # Write CSV: time_s + one column per classifier
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["time_s"] + [f"{n}_global_mean_score"
                                    for n in clf_means]
            writer.writerow(header)
            for i, t in enumerate(t_ref):
                row = [f"{t:.4f}"]
                for mean_arr in clf_means.values():
                    v = mean_arr[i]
                    row.append("" if np.isnan(v) else f"{v:.6f}")
                writer.writerow(row)

        messagebox.showinfo("Save", f"Saved to:\n{path}")
    except Exception as ex:
        messagebox.showerror("Save error", str(ex))


# ── Helper: split features by phase ──────────────────────────────────────────

def _split_by_phase(X, t_centers, calming_intervals, stress_intervals):
    mask_calming = np.zeros(len(t_centers), dtype=bool)
    mask_stress  = np.zeros(len(t_centers), dtype=bool)
    for s, e in calming_intervals:
        mask_calming |= (t_centers >= s) & (t_centers <= e)
    for s, e in stress_intervals:
        mask_stress  |= (t_centers >= s) & (t_centers <= e)
    mask_rest = ~mask_calming & ~mask_stress
    result = {}
    if mask_calming.any(): result["calming"] = X[mask_calming]
    if mask_stress.any():  result["stress"]  = X[mask_stress]
    if mask_rest.any():    result["rest"]    = X[mask_rest]
    return result


# ── Boxplots ──────────────────────────────────────────────────────────────────

def _plot_boxplots_big(phase_features, signal_name):
    phase_labels = list(phase_features.keys())
    colors       = ["#5599dd", "#55bb77", "#8877dd", "#dd5555"]
    feat_names   = ["Mean", "Std", "RMS", "PtP"]
    feat_dim     = 4

    fig, axs = plt.subplots(1, feat_dim, figsize=(4 * feat_dim, 4))
    fig.suptitle(f"Feature Distribution — {signal_name}",
                 fontsize=13, fontweight="bold")

    for fi, feat_name in enumerate(feat_names):
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

        all_vals = np.concatenate([d for d in data_per_phase if len(d) > 0])
        if len(all_vals) > 0:
            ax.set_ylim(np.percentile(all_vals, 2),
                        np.percentile(all_vals, 98))

        ax.set_xticks(range(1, len(phase_labels) + 1))
        ax.set_xticklabels(phase_labels, fontsize=11, rotation=15)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(feat_name, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ── RR interval from ECG ─────────────────────────────────────────────────────

def _compute_rr_from_ecg(ecg, fs):
    from scipy.signal import find_peaks, butter, filtfilt
    ecg = np.asarray(ecg, dtype=float).ravel()
    nyq = fs / 2.0
    b, a = butter(2, [0.5/nyq, min(40.0/nyq, 0.99)], btype="band")
    ecg_f = filtfilt(b, a, ecg)
    peak_val = np.max(np.abs(ecg_f))
    if peak_val == 0:
        return np.array([]), np.array([])
    ecg_n = ecg_f / peak_val
    peaks, _ = find_peaks(ecg_n, height=0.3, distance=int(0.4 * fs))
    if len(peaks) < 2:
        return np.array([]), np.array([])
    rr_ms = (np.diff(peaks) / fs) * 1000.0
    t_rr  = (peaks[:-1] + peaks[1:]) / 2.0 / fs
    valid = (rr_ms >= 240) & (rr_ms <= 2000)
    return t_rr[valid], rr_ms[valid]


def _plot_ecg_rr(ecg, fs, stress_intervals, calming_intervals):
    t_rr, rr_ms = _compute_rr_from_ecg(ecg, fs)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    if len(t_rr) == 0:
        ax.text(0.5, 0.5, "No R-peaks detected.\nCheck ECG signal.",
                ha="center", va="center", fontsize=10, color="#cc4400")
        ax.set_title("RR Interval from ECG", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig
    k = min(11, len(rr_ms))
    rr_smooth = np.convolve(rr_ms, np.ones(k)/k, mode="same")
    rr_mean, rr_std = float(np.mean(rr_ms)), float(np.std(rr_ms))
    thr_low, thr_high = rr_mean - rr_std, rr_mean + rr_std
    ax.scatter(t_rr, rr_ms, s=6, color="#9933aa", alpha=0.35, zorder=2,
               label="Instantaneous RR (ms)")
    ax.plot(t_rr, rr_smooth, color="#660099", lw=1.8, zorder=4,
            label=f"Smoothed RR ({k}-beat avg)")
    ax.fill_between(t_rr, rr_mean, rr_smooth,
                    where=rr_smooth < rr_mean, color="#ffaa44", alpha=0.20)
    ax.fill_between(t_rr, rr_mean, rr_smooth,
                    where=rr_smooth >= rr_mean, color="#aaddff", alpha=0.15)
    ax.axhline(rr_mean,  color="#555555", lw=1.0, ls="--",
               label=f"Mean {rr_mean:.0f} ms")
    ax.axhline(thr_low,  color="#cc0000", lw=1.0, ls=":",
               label=f"−1σ {thr_low:.0f} ms (stress thr.)")
    ax.axhline(thr_high, color="#0055cc", lw=1.0, ls=":",
               label=f"+1σ {thr_high:.0f} ms (relax thr.)")
    in_stress, seg_start = False, 0.0
    for i in range(len(t_rr)):
        if rr_smooth[i] < thr_low and not in_stress:
            seg_start, in_stress = t_rr[i], True
        elif rr_smooth[i] >= thr_low and in_stress:
            ax.axvspan(seg_start, t_rr[i], color="#ffcccc", alpha=0.55)
            in_stress = False
    if in_stress:
        ax.axvspan(seg_start, t_rr[-1], color="#ffcccc", alpha=0.55)
    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.22)
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.18)
    ax.set_ylim(max(240, rr_mean - 4*rr_std), min(2000, rr_mean + 4*rr_std))
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("RR interval (ms)", fontsize=9)
    ax.set_title("RR Interval from ECG — red zone: short RR = fast HR = potential stress",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.75, ncol=2)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


# ── Respiratory rate from RESP ────────────────────────────────────────────────

def _compute_resp_rate(resp, fs):
    from scipy.signal import find_peaks, butter, filtfilt
    resp = np.asarray(resp, dtype=float).ravel()
    nyq  = fs / 2.0
    b, a = butter(2, [max(0.1/nyq, 1e-4), min(0.8/nyq, 0.99)], btype="band")
    resp_f = filtfilt(b, a, resp)
    peak_val = np.max(np.abs(resp_f))
    if peak_val == 0:
        return np.array([]), np.array([])
    resp_n = resp_f / peak_val
    peaks, _ = find_peaks(resp_n, height=0.2, distance=int(1.5*fs))
    if len(peaks) < 2:
        return np.array([]), np.array([])
    rpm   = 60.0 / (np.diff(peaks) / fs)
    t_rpm = (peaks[:-1] + peaks[1:]) / 2.0 / fs
    valid = (rpm >= 1) & (rpm <= 60)
    return t_rpm[valid], rpm[valid]


def _plot_resp_rate(resp, fs, stress_intervals, calming_intervals):
    t_rpm, rpm = _compute_resp_rate(resp, fs)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    if len(t_rpm) == 0:
        ax.text(0.5, 0.5, "No respiratory cycles detected.\nCheck RESP signal.",
                ha="center", va="center", fontsize=10, color="#cc4400")
        ax.set_title("Respiratory Rate (rpm) from RESP", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig
    k = min(7, len(rpm))
    rpm_smooth = np.convolve(rpm, np.ones(k)/k, mode="same")
    rpm_mean, rpm_std = float(np.mean(rpm)), float(np.std(rpm))
    thr_high, thr_low = rpm_mean + rpm_std, rpm_mean - rpm_std
    ax.scatter(t_rpm, rpm, s=6, color="#2288cc", alpha=0.35, zorder=2,
               label="Instantaneous rpm")
    ax.plot(t_rpm, rpm_smooth, color="#115588", lw=1.8, zorder=4,
            label=f"Smoothed ({k}-cycle avg)")
    ax.fill_between(t_rpm, rpm_mean, rpm_smooth,
                    where=rpm_smooth >= rpm_mean, color="#ffaa44", alpha=0.20)
    ax.fill_between(t_rpm, rpm_mean, rpm_smooth,
                    where=rpm_smooth < rpm_mean, color="#88ccff", alpha=0.15)
    ax.axhline(rpm_mean, color="#555555", lw=1.0, ls="--",
               label=f"Mean {rpm_mean:.1f} rpm")
    ax.axhline(thr_high, color="#cc0000", lw=1.0, ls=":",
               label=f"+1σ {thr_high:.1f} rpm")
    ax.axhline(thr_low,  color="#0055cc", lw=1.0, ls=":",
               label=f"−1σ {thr_low:.1f} rpm")
    ax.axhspan(12, 20, color="#ddffdd", alpha=0.25, label="Normal (12–20 rpm)")
    in_stress, seg_start = False, 0.0
    for i in range(len(t_rpm)):
        if rpm_smooth[i] > thr_high and not in_stress:
            seg_start, in_stress = t_rpm[i], True
        elif rpm_smooth[i] <= thr_high and in_stress:
            ax.axvspan(seg_start, t_rpm[i], color="#ffcccc", alpha=0.55)
            in_stress = False
    if in_stress:
        ax.axvspan(seg_start, t_rpm[-1], color="#ffcccc", alpha=0.55)
    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.22)
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.18)
    ax.set_ylim(max(0, rpm_mean - 4*rpm_std), min(60, rpm_mean + 4*rpm_std))
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("rpm", fontsize=9)
    ax.set_title("Respiratory Rate (rpm) from RESP — red zone: above stress threshold",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.75, ncol=2)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


# ── Global tab plots ──────────────────────────────────────────────────────────

def _plot_global_stress_probability(stress_map, stress_intervals,
                                     calming_intervals, model_name=""):
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    # Mismo area de dibujo que el heatmap: [_LEFT, _RIGHT]
    ax  = fig.add_axes([_LEFT, 0.18, _RIGHT - _LEFT, 0.72])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(stress_map), 1)))
    all_scores = []
    for (sig_name, (t_c, preds, scores)), color in zip(stress_map.items(), colors):
        ax.plot(t_c, scores, lw=0.9, alpha=0.55, color=color, label=sig_name)
        all_scores.append((t_c, scores))
    if len(all_scores) > 1:
        t_ref = max(all_scores, key=lambda x: len(x[0]))[0]
        interp = np.array([np.interp(t_ref, t, s, left=np.nan, right=np.nan)
                           for t, s in all_scores])
        mean_score = np.nanmean(interp, axis=0)
        ax.plot(t_ref, mean_score, lw=2.2, color="black", label="Mean", zorder=5)
        ax.fill_between(t_ref, mean_score, alpha=0.15, color="black")
    ax.axhline(0.5, color="#cc0000", lw=0.9, ls="--", label="Threshold 0.5")
    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.30)
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.30)
    title = "Combined stress probability — all signals"
    if model_name: title += f"  [{model_name}]"
    ax.set_ylim(0, 1.05)

    # ── Mismo rango temporal que el heatmap (hasta el final REAL del registro) ─
    t_arrays = [v[0] for v in stress_map.values() if len(v[0]) > 0]
    if t_arrays:
        t_ref_full = max(t_arrays, key=len)
        t0    = float(t_ref_full[0])
        t_end = float(t_ref_full[-1])
        if len(t_ref_full) > 1:
            # el ultimo valor es el CENTRO de la ultima ventana: se anade
            # media ventana para llegar al final real del registro
            t_end += float(t_ref_full[-1] - t_ref_full[-2]) / 2.0
        ax.set_xlim(t0, t_end)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("P(stress)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    # Leyenda en el margen derecho (misma zona que la colorbar del heatmap),
    # de modo que el area de dibujo mantiene el mismo ancho [_LEFT, _RIGHT].
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              borderaxespad=0, framealpha=0.75)
    ax.grid(True, alpha=0.25)
    return fig


def _plot_global_heatmap_mean(stress_map, stress_intervals,
                               calming_intervals, model_name=""):
    import scipy.signal

    sig_names = list(stress_map.keys())
    if not sig_names:
        fig = plt.figure(figsize=(_FIG_W, _FIG_H))
        ax  = fig.add_axes([_LEFT, 0.18, _RIGHT - _LEFT, 0.72])
        ax.text(0.5, 0.5, "No results.", ha="center", va="center")
        return fig

    t_ref = max((stress_map[s][0] for s in sig_names), key=len)
    t_min = float(t_ref[0])
    # El ultimo t es el CENTRO de la ultima ventana; se anade media ventana
    # para que la representacion llegue al final real del registro (sin margen).
    t_max = float(t_ref[-1])
    if len(t_ref) > 1:
        t_max += float(t_ref[-1] - t_ref[-2]) / 2.0

    rows = []
    for s in sig_names:
        t, _, scores = stress_map[s]
        if len(t) < 2:
            continue
        rows.append(np.interp(t_ref, t, scores, left=np.nan, right=np.nan))
    rows = np.array(rows)

    mean_prob = np.nanmean(rows, axis=0)
    mean_prob = np.nan_to_num(mean_prob, nan=np.nanmean(mean_prob))

    analytic_signal = scipy.signal.hilbert(mean_prob)
    envelope = np.abs(analytic_signal)

    window = min(21, len(envelope))
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        envelope = scipy.signal.savgol_filter(envelope, window_length=window, polyorder=3)

    envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-12)

    # ── Layout: mismo area [_LEFT, _RIGHT] que la grafica de probabilidad ──────
    HM_W = _RIGHT - _LEFT

    fig      = plt.figure(figsize=(_FIG_W, _FIG_H))
    ax_heat  = fig.add_axes([_LEFT, 0.32, HM_W, 0.52])
    ax_phase = fig.add_axes([_LEFT, 0.10, HM_W, 0.18])
    ax_cbar  = fig.add_axes([_CBAR_LEFT, 0.32, _CBAR_W, 0.52])

    im = ax_heat.imshow(
        envelope[np.newaxis, :],
        aspect="auto",
        cmap=STRESS_CMAP,
        vmin=0.0,
        vmax=1.0,
        extent=[t_min, t_max, 0.5, -0.5],
        interpolation="bilinear",
    )

    ax_heat.set_yticks([0])
    ax_heat.set_yticklabels(["Envelope"], fontsize=9)
    ax_heat.set_xticks([])

    title = "Stress probability envelope heatmap"
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
    cbar.set_label("Normalized envelope", fontsize=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)

    ax_phase.set_xlim(t_min, t_max)
    ax_phase.set_ylim(0, 1)
    ax_phase.set_yticks([])
    ax_phase.set_xlabel("Time (s)", fontsize=9)
    ax_phase.axhspan(0, 1, color="#dddddd", alpha=0.4)
    for s, e in calming_intervals:
        ax_phase.axvspan(s, e, color="#b6f0c8", alpha=0.9)
        ax_phase.text((s+e)/2, 0.5, "Calming", ha="center", va="center",
                      fontsize=7, color="#1a6b3a", fontweight="bold")
    for s, e in stress_intervals:
        ax_phase.axvspan(s, e, color="#ffb3b3", alpha=0.9)
        ax_phase.text((s+e)/2, 0.5, "Stress", ha="center", va="center",
                      fontsize=7, color="#aa2222", fontweight="bold")
    for spine in ["top", "right", "left"]:
        ax_phase.spines[spine].set_visible(False)
    return fig


# ── Per-signal plot functions ─────────────────────────────────────────────────

def _plot_stress_timeline(t_centers, preds, scores, sig_name,
                           stress_intervals, model_name=""):
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
    if model_name: title += f"  [{model_name}]"
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

def _open_popup(fig, title):
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
        ("RR Interval from ECG", "rr_ecg"),
        ("Resp. rate from RESP", "resp_rate"),
    ]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._figs: list[plt.Figure] = []
        self._signals:           dict = {}
        self._fs_map:            dict = {}
        self._stress_intervals:  list = []
        self._calming_intervals: list = []
        self._all_stress_maps:   dict = {}
        self._sig_vars:   dict = {}
        self._plot_vars:  dict = {}
        self._clf_vars:   dict = {}
        self._window_var  = None
        self._step_var    = None
        self._status_var  = None
        self._results_frame = None
        self._build_layout()

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._ctrl = ctk.CTkScrollableFrame(self, width=230)
        self._ctrl.grid(row=0, column=0, sticky="ns", padx=(6,2), pady=6)
        ctk.CTkLabel(self._ctrl, text="Analysis controls",
                     font=("Arial", 13, "bold")).pack(anchor="w", pady=(8,4))
        self._build_controls()
        self._results_frame = ctk.CTkFrame(self)
        self._results_frame.grid(row=0, column=1, sticky="nsew",
                                  padx=(2,6), pady=6)
        self._results_frame.grid_columnconfigure(0, weight=1)
        self._results_frame.grid_rowconfigure(0, weight=1)
        ctk.CTkLabel(self._results_frame,
                     text="Configure the controls on the left and press Run.",
                     text_color="gray", font=("Arial", 13)).pack(pady=40)

    def _build_controls(self):
        ctrl = self._ctrl
        self._sig_section = ctk.CTkFrame(ctrl, fg_color="transparent")
        self._sig_section.pack(fill="x", pady=(0,6))
        ctk.CTkLabel(self._sig_section, text="Signals to analyse",
                     font=("Arial", 11, "bold")).pack(anchor="w")
        ctk.CTkLabel(self._sig_section, text="(load a subject first)",
                     text_color="gray", font=("Arial", 10)).pack(anchor="w")

        ctk.CTkLabel(ctrl, text="Classifiers",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(10,2))
        self._clf_vars = {}
        defaults = {"Random Forest": True, "LightGBM": False,
                    "SVM": False, "KNN": False}
        for clf_name in CLASSIFIERS:
            var = ctk.BooleanVar(value=defaults.get(clf_name, False))
            self._clf_vars[clf_name] = var
            ctk.CTkCheckBox(ctrl, text=clf_name, variable=var,
                            font=("Arial", 11)).pack(anchor="w", padx=4, pady=1)
        if not CLASSIFIERS:
            ctk.CTkLabel(ctrl, text="No classifiers available.",
                         text_color="#cc4400", font=("Arial", 10)).pack(anchor="w", padx=4)

        ctk.CTkLabel(ctrl, text="Plot types",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(10,2))
        self._plot_vars = {}
        for label, key in self.PLOT_TYPES:
            var = ctk.BooleanVar(value=True)
            self._plot_vars[key] = var
            ctk.CTkCheckBox(ctrl, text=label, variable=var,
                            font=("Arial", 11)).pack(anchor="w", padx=4, pady=1)

        ctk.CTkLabel(ctrl, text="Window size (s)",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(10,2))
        self._window_var = ctk.StringVar(value=str(RF_WINDOW_SEC))
        ctk.CTkEntry(ctrl, textvariable=self._window_var, height=28).pack(fill="x", padx=4)

        ctk.CTkLabel(ctrl, text="Step size (s)",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(6,2))
        self._step_var = ctk.StringVar(value=str(RF_STEP_SEC))
        ctk.CTkEntry(ctrl, textvariable=self._step_var, height=28).pack(fill="x", padx=4)

        ctk.CTkButton(ctrl, text="▶  Run analysis",
                      height=38, fg_color="#336699", hover_color="#224477",
                      command=self._run).pack(fill="x", padx=4, pady=(14,4))

        self._status_var = ctk.StringVar(value="")
        ctk.CTkLabel(ctrl, textvariable=self._status_var,
                     text_color="gray", font=("Arial", 10),
                     wraplength=210).pack(padx=4, pady=2)

    def load_data(self, signals, fs_map, stress_intervals, calming_intervals):
        self._signals           = signals
        self._fs_map            = fs_map
        self._stress_intervals  = stress_intervals
        self._calming_intervals = calming_intervals
        self._refresh_signal_checkboxes()

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

    def _selected_signals(self):
        return [s for s, v in self._sig_vars.items() if v.get()]

    def _selected_plots(self):
        return {k for k, v in self._plot_vars.items() if v.get()}

    def _selected_classifiers(self):
        return [n for n, v in self._clf_vars.items() if v.get()]

    def _get_window_step(self):
        try:
            win  = float(self._window_var.get())
            step = float(self._step_var.get())
            assert win > 0 and step > 0 and step <= win
            return win, step
        except Exception:
            return RF_WINDOW_SEC, RF_STEP_SEC

    def _set_status(self, msg):
        if self._status_var:
            self._status_var.set(msg)
        self.update_idletasks()

    def _clear_results(self):
        for fig in self._figs:
            try:
                plt.close(fig)
            except Exception:
                pass
        self._figs.clear()
        plt.close("all")
        self._all_stress_maps = {}
        children = list(self._results_frame.winfo_children())
        def _do_destroy():
            for w in children:
                try:
                    if w.winfo_exists():
                        w.destroy()
                except Exception:
                    pass
        self._results_frame.after_idle(_do_destroy)
        self._ctrl.grid()
        self.grid_columnconfigure(0, weight=0)

    def reset_for_new_subject(self):
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
        lbl = f"  Stressed windows: {n_stressed} / {n_total}   ·   Mean prob: {scores.mean():.2f}"
        frame = ctk.CTkFrame(parent, fg_color="#336699", corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=6, pady=(8,2))
        ctk.CTkLabel(frame, text=lbl, font=("Arial", 11, "bold"),
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
            ctk.CTkLabel(header, text=title, font=("Arial", 11, "bold"),
                         anchor="w").grid(row=0, column=0, sticky="w", padx=4)
        popup_title = title or "Plot"
        ctk.CTkButton(header, text="⛶  Expand", width=90, height=24,
                      font=("Arial", 10), fg_color="#4526FA", hover_color="#A3A3A3",
                      command=lambda f=fig, t=popup_title: _open_popup(f, t)
                      ).grid(row=0, column=1, sticky="e", padx=4)
        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="ew")

    def _add_section_label(self, parent, text, row):
        ctk.CTkLabel(parent, text=text, font=("Arial", 12, "bold"),
                     anchor="w").grid(row=row, column=0, sticky="ew",
                                      padx=8, pady=(10,2))

    def _add_collapsible_clf_section(self, parent, clf_name, start_row):
        is_expanded   = ctk.BooleanVar(value=True)
        content_frame = ctk.CTkFrame(parent, fg_color="transparent")
        content_frame.grid(row=start_row + 1, column=0, sticky="ew", padx=0, pady=0)
        content_frame.grid_columnconfigure(0, weight=1)

        def _toggle():
            if is_expanded.get():
                content_frame.grid_remove()
                is_expanded.set(False)
                btn.configure(text=f"▶  {clf_name}")
            else:
                content_frame.grid()
                is_expanded.set(True)
                btn.configure(text=f"▼  {clf_name}")

        btn = ctk.CTkButton(parent, text=f"▼  {clf_name}",
                            font=("Arial", 14, "bold"),
                            fg_color="#224466", hover_color="#336699",
                            anchor="w", command=_toggle)
        btn.grid(row=start_row, column=0, sticky="ew", padx=6, pady=(10,2))
        return content_frame, start_row + 2

    def _ask_export_csv(self):
        if not self._all_stress_maps:
            return
        if messagebox.askyesno(
            "Save CSV",
            "Save .csv with global stress probability\n"
            "for the selected models?",
        ):
            _export_stress_csv(self._all_stress_maps, self)

    def _run(self):
        self._clear_results()
        selected_sigs  = self._selected_signals()
        selected_plots = self._selected_plots()
        selected_clfs  = self._selected_classifiers()
        window_sec, step_sec = self._get_window_step()

        if not selected_sigs:
            self._set_status("Select at least one signal."); return
        if not selected_plots:
            self._set_status("Select at least one plot type."); return
        if not self._signals:
            self._set_status("Load a subject first."); return
        if not selected_clfs:
            self._set_status("Select at least one classifier."); return

        self._set_status("Running…")
        self.update_idletasks()

        signals = {s: self._signals[s] for s in selected_sigs if s in self._signals}
        t_phys, y_phys = build_physiological_labels(
            signals, self._fs_map, window_sec, step_sec)

        for w in list(self._results_frame.winfo_children()):
            try:
                if w.winfo_exists(): w.destroy()
            except Exception:
                pass

        tab_view = ctk.CTkTabview(self._results_frame)
        tab_view.pack(fill="both", expand=True, padx=2, pady=2)

        # Signal tabs
        tabs: dict = {}
        for sig_name in selected_sigs:
            if sig_name in signals:
                tab_view.add(sig_name)
                scroll = ctk.CTkScrollableFrame(tab_view.tab(sig_name))
                scroll.pack(fill="both", expand=True)
                scroll.grid_columnconfigure(0, weight=1)
                tabs[sig_name] = scroll

        # Global tab
        tab_view.add("Global")
        global_scroll = ctk.CTkScrollableFrame(tab_view.tab("Global"))
        global_scroll.pack(fill="both", expand=True)
        global_scroll.grid_columnconfigure(0, weight=1)

        tab_rows = {sig: 0 for sig in tabs}
        g_row, n_done = 0, 0

        # ── "Gráficas generales" collapsible section in Global (always first) ──
        gen_content, g_row = self._add_collapsible_clf_section(
            global_scroll, "General plots", g_row
        )
        gen_content.grid_columnconfigure(0, weight=1)
        gen_inner = 0

        # RR and RESP go inside "Gráficas generales"
        if "rr_ecg" in selected_plots and "ECG" in signals:
            fig_rr = _plot_ecg_rr(
                signals["ECG"], self._fs_map.get("ECG", 2000),
                self._stress_intervals, self._calming_intervals)
            self._add_plot_in(gen_content, "RR Interval from ECG", fig_rr, gen_inner)
            gen_inner += 1

        if "resp_rate" in selected_plots and "RESP" in signals:
            fig_resp = _plot_resp_rate(
                signals["RESP"], self._fs_map.get("RESP", 2000),
                self._stress_intervals, self._calming_intervals)
            self._add_plot_in(gen_content, "Respiratory Rate (rpm) from RESP",
                               fig_resp, gen_inner)
            gen_inner += 1

        if gen_inner == 0:
            ctk.CTkLabel(gen_content, text="No general plots selected.",
                         text_color="gray", font=("Arial", 11)).grid(
                row=0, column=0, padx=8, pady=8, sticky="w")

        # ── Per-classifier loop ───────────────────────────────────────────────
        total_steps = len(selected_clfs) * len(selected_sigs)
        current_step = 0

        for clf_idx, clf_name in enumerate(selected_clfs):
            self._set_status(f"Running {clf_name}… ({int(current_step/total_steps*100)}%)")
            self.update_idletasks()

            stress_map: dict = {}
            per_sig:    dict = {}

            for sig_idx, sig_name in enumerate(selected_sigs):
                if sig_name not in signals:
                    current_step += 1
                    continue
                current_step += 1
                pct = int(current_step / total_steps * 100)
                self._set_status(f"Running {clf_name}… ({pct}%)")
                self.update_idletasks()
                fs = self._fs_map.get(sig_name, 2000)
                X_all, t_centers = extract_features_windowed(
                    signals[sig_name], fs, sig_name, window_sec, step_sec)
                if len(X_all) == 0: continue
                n_win = min(len(X_all), len(t_phys))
                if n_win == 0: continue
                X_train = X_all[:n_win]
                y_train = y_phys[:n_win]
                if len(np.unique(y_train)) < 2: continue
                try:
                    clf = _build_classifier(clf_name)
                    preds, scores = _fit_predict(clf, X_train, y_train, X_all)
                except Exception as ex:
                    self._set_status(f"{clf_name} failed: {ex}"); continue

                stress_map[sig_name] = (t_centers, preds, scores)
                per_sig[sig_name] = {
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

            if not stress_map: continue
            self._all_stress_maps[clf_name] = stress_map

            # Fill per-signal tabs
            for sig_name, res in per_sig.items():
                if sig_name not in tabs: continue
                parent    = tabs[sig_name]
                t_centers = res["t_centers"]
                preds     = res["preds"]
                scores    = res["scores"]
                row       = tab_rows[sig_name]
                fs        = self._fs_map.get(sig_name, 2000)

                content, row = self._add_collapsible_clf_section(parent, clf_name, row)
                inner_row = 0
                content.grid_columnconfigure(0, weight=1)

                self._add_section_header_in(content, sig_name, preds, scores, inner_row)
                inner_row += 1

                if "boxplots" in selected_plots:
                    fig = _plot_boxplots_big(res["phase_feats"], sig_name)
                    self._add_plot_in(content, f"Feature distributions — {clf_name}", fig, inner_row)
                    inner_row += 1

                if "roc" in selected_plots:
                    fig = plot_roc(res["y_session"], scores, sig_name)
                    self._add_plot_in(content, f"ROC curve — {clf_name}", fig, inner_row)
                    inner_row += 1

                if "timeline" in selected_plots:
                    fig = _plot_stress_timeline(t_centers, preds, scores, sig_name,
                                                self._stress_intervals, clf_name)
                    self._add_plot_in(content, f"Stress timeline — {clf_name}", fig, inner_row)
                    inner_row += 1

                # RR and RESP only in "Gráficas generales", not per signal tabs
                tab_rows[sig_name] = row

            # Global tab: collapsible per classifier
            content_g, g_row = self._add_collapsible_clf_section(
                global_scroll, clf_name, g_row)
            content_g.grid_columnconfigure(0, weight=1)
            inner_g = 0

            self._add_section_label(
                content_g, f"Combined stress probability — {clf_name}", inner_g)
            inner_g += 1
            fig_prob = _plot_global_stress_probability(
                stress_map, self._stress_intervals,
                self._calming_intervals, clf_name)
            self._add_plot_in(content_g, "", fig_prob, inner_g)
            inner_g += 1

            self._add_section_label(
                content_g, f"Mean stress heatmap — {clf_name}", inner_g)
            inner_g += 1
            fig_heat = _plot_global_heatmap_mean(
                stress_map, self._stress_intervals,
                self._calming_intervals, clf_name)
            self._add_plot_in(content_g, "", fig_heat, inner_g)

            n_done += 1

        if n_done == 0:
            ctk.CTkLabel(self._results_frame,
                         text="Not enough data or all classifiers failed.",
                         text_color="#cc4400", font=("Arial", 12)).pack(pady=20)
            self._set_status("No results.")
            return

        self._set_status(f"Done — {n_done} classifier(s), {len(tabs)} signal(s) analysed.")
        self._ctrl.grid_remove()
        self.grid_columnconfigure(0, weight=0)
        self.after(300, self._ask_export_csv)