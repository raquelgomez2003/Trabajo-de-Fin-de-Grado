"""
analysis_window.py
Analysis tab — stress prediction with configurable plot selection.
"""

from __future__ import annotations
import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Shared layout constants so both plots have identical margins
_FIG_W      = 12.0   # inches — common figure width
_FIG_H      = 3.5    # inches — common figure height
_LEFT       = 0.07   # fraction — left margin (room for y-label)
_RIGHT      = 0.87   # fraction — right edge (room for colorbar / legend)
_CBAR_LEFT  = 0.88   # fraction — colorbar left edge
_CBAR_W     = 0.02   # fraction — colorbar width


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


# ── Global tab plots ──────────────────────────────────────────────────────────

def _plot_global_stress_probability(
    stress_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stress_intervals: list[tuple[float, float]],
    calming_intervals: list[tuple[float, float]],
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

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("P(stress)", fontsize=10)
    ax.set_title("Combined stress probability — all signals",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              framealpha=0.75)
    ax.grid(True, alpha=0.25)
    return fig


def _plot_global_heatmap_mean(
    stress_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stress_intervals: list[tuple[float, float]],
    calming_intervals: list[tuple[float, float]],
) -> plt.Figure:
    """
    Single-row heatmap of MEAN stress probability.
    Identical figure size and horizontal margins as the probability plot.
    """
    sig_names = list(stress_map.keys())

    if not sig_names:
        fig = plt.figure(figsize=(_FIG_W, _FIG_H))
        ax  = fig.add_axes([_LEFT, 0.18, _RIGHT - _LEFT, 0.72])
        ax.text(0.5, 0.5, "No results to display.",
                ha="center", va="center")
        return fig

    # Common time grid
    t_ref        = max((stress_map[s][0] for s in sig_names), key=len)
    t_min, t_max = t_ref[0], t_ref[-1]

    # Mean score across signals
    rows = []
    for sig in sig_names:
        t_sig, _, scores = stress_map[sig]
        if len(t_sig) > 1:
            rows.append(
                np.interp(t_ref, t_sig, scores, left=np.nan, right=np.nan)
            )
    mean_row = np.nanmean(np.array(rows), axis=0)

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))

    # Heatmap axes — same left/right as probability plot
    ax_heat  = fig.add_axes([_LEFT, 0.32, _RIGHT - _LEFT, 0.52])
    # Phase bar axes — thin strip below heatmap, same left/right
    ax_phase = fig.add_axes([_LEFT, 0.10, _RIGHT - _LEFT, 0.18])
    # Colorbar axes — same right edge as probability legend area
    ax_cbar  = fig.add_axes([_CBAR_LEFT, 0.32, _CBAR_W, 0.52])

    # Heatmap
    im = ax_heat.imshow(
        mean_row[np.newaxis, :],
        aspect="auto",
        cmap=STRESS_CMAP,
        vmin=0.0, vmax=1.0,
        extent=[t_min, t_max, 0.5, -0.5],
        interpolation="bilinear",
    )
    ax_heat.set_yticks([0])
    ax_heat.set_yticklabels(["Mean"], fontsize=9)
    ax_heat.set_xticks([])
    ax_heat.set_title(
        "Mean stress probability heatmap",
        fontsize=11, fontweight="bold", pad=5,
    )

    # Session boundary lines
    for s, e in calming_intervals:
        ax_heat.axvline(s, color="#00aa44", lw=1.0, ls="--", alpha=0.7)
        ax_heat.axvline(e, color="#00aa44", lw=1.0, ls="--", alpha=0.7)
    for s, e in stress_intervals:
        ax_heat.axvline(s, color="#cc0000", lw=1.0, ls="--", alpha=0.7)
        ax_heat.axvline(e, color="#cc0000", lw=1.0, ls="--", alpha=0.7)

    # Colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("P(stress)", fontsize=8)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)

    # Phase bar
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
    ax.set_title(f"Stress probability over time — {sig_name}",
                 fontsize=10, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_rr_detail(
    ecg_signal: np.ndarray,
    fs: float,
    stress_intervals: list[tuple[float, float]],
    calming_intervals: list[tuple[float, float]],
) -> plt.Figure:
    rr_times, rr_ms = compute_rr(ecg_signal, fs)

    fig, ax = plt.subplots(figsize=(11, 3.5))

    if len(rr_times) == 0:
        ax.text(0.5, 0.5,
                "No RR intervals detected — check ECG signal",
                ha="center", va="center")
        ax.set_title("RR interval detail")
        fig.tight_layout()
        return fig

    ax.plot(rr_times, rr_ms, color="#9933aa", lw=0.9, zorder=3,
            label="RR interval")

    rr_mean    = np.mean(rr_ms)
    low_thresh = rr_mean - np.std(rr_ms)
    ax.axhline(rr_mean, color="#777777", lw=1.0, ls="--",
               label=f"Global mean {rr_mean:.0f} ms")
    ax.axhline(low_thresh, color="#cc0000", lw=0.9, ls=":",
               label=f"Stress threshold {low_thresh:.0f} ms")

    mask = rr_ms < low_thresh
    if mask.any():
        ax.scatter(rr_times[mask], rr_ms[mask],
                   color="#cc0000", s=12, zorder=5,
                   label="Short RR (stress)")

    y_top = float(np.nanmax(rr_ms)) + 30 if len(rr_ms) else rr_mean + 50

    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.35, zorder=0)
        phase_mask = (rr_times >= s) & (rr_times <= e)
        if phase_mask.sum() > 2:
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ms[phase_mask]) ** 2)))
            mean  = float(np.mean(rr_ms[phase_mask]))
            ax.text((s + e) / 2, y_top,
                    f"Calming\nmean {mean:.0f} ms\nRMSSD {rmssd:.1f}",
                    ha="center", va="top", fontsize=7, color="#1a6b3a",
                    bbox=dict(fc="white", alpha=0.6, ec="none"))

    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.35, zorder=0)
        phase_mask = (rr_times >= s) & (rr_times <= e)
        if phase_mask.sum() > 2:
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ms[phase_mask]) ** 2)))
            mean  = float(np.mean(rr_ms[phase_mask]))
            ax.text((s + e) / 2, y_top,
                    f"Stress\nmean {mean:.0f} ms\nRMSSD {rmssd:.1f}",
                    ha="center", va="top", fontsize=7, color="#aa2222",
                    bbox=dict(fc="white", alpha=0.6, ec="none"))

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("RR interval (ms)", fontsize=9)
    ax.set_title(
        "RR interval detail — short RR = fast HR = potential stress",
        fontsize=10, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


# ── Popup helper ──────────────────────────────────────────────────────────────

def _open_popup(fig: plt.Figure, title: str) -> None:
    """Open a figure in a resizable popup window with a navigation toolbar."""
    popup = ctk.CTkToplevel()
    popup.title(title)
    popup.geometry("1100x520")
    popup.grab_set()          # modal — blocks the main window until closed

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
        ("Boxplots",        "boxplots"),
        ("ROC curve",       "roc"),
        ("Stress timeline", "timeline"),
        ("RR detail",       "rr_detail"),
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
        self._ctrl.grid(row=0, column=0, sticky="ns",
                        padx=(6, 2), pady=6)
        ctk.CTkLabel(
            self._ctrl, text="Analysis controls",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(8, 4))
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

        self._sig_section = ctk.CTkFrame(ctrl, fg_color="transparent")
        self._sig_section.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            self._sig_section, text="Signals to analyse",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            self._sig_section, text="(load a subject first)",
            text_color="gray", font=("Arial", 10),
        ).pack(anchor="w")

        ctk.CTkLabel(
            ctrl, text="Plot types",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", pady=(8, 2))
        self._plot_vars = {}
        for label, key in self.PLOT_TYPES:
            var = ctk.BooleanVar(value=True)
            self._plot_vars[key] = var
            ctk.CTkCheckBox(
                ctrl, text=label, variable=var,
                font=("Arial", 11),
            ).pack(anchor="w", padx=4, pady=1)

        ctk.CTkLabel(
            ctrl, text="Window size (s)",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", pady=(10, 2))
        self._window_var = ctk.StringVar(value=str(RF_WINDOW_SEC))
        ctk.CTkEntry(
            ctrl, textvariable=self._window_var, height=28,
        ).pack(fill="x", padx=4)

        ctk.CTkLabel(
            ctrl, text="Step size (s)",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", pady=(6, 2))
        self._step_var = ctk.StringVar(value=str(RF_STEP_SEC))
        ctk.CTkEntry(
            ctrl, textvariable=self._step_var, height=28,
        ).pack(fill="x", padx=4)

        ctk.CTkButton(
            ctrl, text="▶  Run analysis",
            height=38, fg_color="#336699", hover_color="#224477",
            command=self._run,
        ).pack(fill="x", padx=4, pady=(14, 4))

        self._status_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            ctrl, textvariable=self._status_var,
            text_color="gray", font=("Arial", 10),
            wraplength=210,
        ).pack(padx=4, pady=2)

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
        ctk.CTkLabel(
            self._sig_section, text="Signals to analyse",
            font=("Arial", 11, "bold"),
        ).pack(anchor="w")
        self._sig_vars = {}
        for sig in self._signals:
            var = ctk.BooleanVar(value=True)
            self._sig_vars[sig] = var
            ctk.CTkCheckBox(
                self._sig_section, text=sig, variable=var,
                font=("Arial", 11),
            ).pack(anchor="w", padx=4, pady=1)
        if self._status_var:
            self._status_var.set(f"{len(self._signals)} signal(s) ready.")

    def _selected_signals(self) -> list[str]:
        return [s for s, v in self._sig_vars.items() if v.get()]

    def _selected_plots(self) -> set[str]:
        return {k for k, v in self._plot_vars.items() if v.get()}

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
        for w in self._results_frame.winfo_children():
            w.destroy()
        for fig in self._figs:
            plt.close(fig)
        self._figs.clear()
        # Restore control panel
        self._ctrl.grid()
        self.grid_columnconfigure(0, weight=0)

    def _add_section_header_in(
        self,
        parent: ctk.CTkScrollableFrame,
        sig_name: str,
        preds: np.ndarray,
        scores: np.ndarray,
        row: int,
    ) -> None:
        n_stressed = int(preds.sum())
        n_total    = len(preds)

        frame = ctk.CTkFrame(parent, fg_color="#336699", corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=6, pady=(8, 2))
        ctk.CTkLabel(
            frame,
            text=(
                f"  Stressed windows: {n_stressed} / {n_total}   ·   "
                f"Mean stress prob: {scores.mean():.2f}"
            ),
            font=("Arial", 11, "bold"),
            text_color="white",
        ).pack(side="left", padx=8, pady=6)

    def _add_plot_in(
        self,
        parent: ctk.CTkScrollableFrame,
        title: str,
        fig: plt.Figure,
        row: int,
    ) -> None:
        """Embed a matplotlib figure with an expand button."""
        self._figs.append(fig)

        outer = ctk.CTkFrame(parent, fg_color="transparent")
        outer.grid(row=row, column=0, sticky="ew", padx=6, pady=4)
        outer.grid_columnconfigure(0, weight=1)

        # Title row with expand button
        header = ctk.CTkFrame(outer, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        if title:
            ctk.CTkLabel(
                header, text=title,
                font=("Arial", 11, "bold"), anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=4)

        # Expand button — opens popup
        popup_title = title or "Plot"
        ctk.CTkButton(
            header,
            text="⛶  Expand",
            width=90, height=24,
            font=("Arial", 10),
            fg_color="#4526FA", hover_color="#A3A3A3",
            command=lambda f=fig, t=popup_title: _open_popup(f, t),
        ).grid(row=0, column=1, sticky="e", padx=4)

        # Canvas
        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="ew")

    def _add_section_label(
        self,
        parent: ctk.CTkScrollableFrame,
        text: str,
        row: int,
    ) -> None:
        ctk.CTkLabel(
            parent, text=text,
            font=("Arial", 12, "bold"), anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=8, pady=(10, 2))

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self._clear_results()

        selected_sigs        = self._selected_signals()
        selected_plots       = self._selected_plots()
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

        self._set_status("Running…")
        self.update_idletasks()

        signals = {
            s: self._signals[s]
            for s in selected_sigs
            if s in self._signals
        }

        t_phys, y_phys = build_physiological_labels(
            signals, self._fs_map, window_sec, step_sec
        )

        stress_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        per_sig_results: dict[str, dict] = {}

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

            model  = StressModel()
            model.fit(X_train, y_train)
            preds  = model.predict(X_all)
            scores = model.predict_proba(X_all)

            stress_map[sig_name] = (t_centers, preds, scores)
            per_sig_results[sig_name] = {
                "X_all":       X_all,
                "t_centers":   t_centers,
                "preds":       preds,
                "scores":      scores,
                "model":       model,
                "y_session":   build_labels_from_intervals(
                                   t_centers, self._stress_intervals),
                "phase_feats": _split_by_phase(
                                   X_all, t_centers,
                                   self._calming_intervals,
                                   self._stress_intervals),
            }

        if not stress_map:
            ctk.CTkLabel(
                self._results_frame,
                text=(
                    f"Not enough data.\n"
                    f"Signals need ≥ 2× window size ({window_sec:.0f} s)."
                ),
                text_color="#cc4400", font=("Arial", 12),
            ).pack(pady=20)
            self._set_status("No results — check window size.")
            return {}

        # ── Build tab view ────────────────────────────────────────────────────
        tab_view = ctk.CTkTabview(self._results_frame)
        tab_view.pack(fill="both", expand=True, padx=2, pady=2)

        tabs: dict[str, ctk.CTkScrollableFrame] = {}
        for sig_name in per_sig_results:
            tab_view.add(sig_name)
            scroll = ctk.CTkScrollableFrame(tab_view.tab(sig_name))
            scroll.pack(fill="both", expand=True)
            scroll.grid_columnconfigure(0, weight=1)
            tabs[sig_name] = scroll

        tab_view.add("Global")
        global_scroll = ctk.CTkScrollableFrame(tab_view.tab("Global"))
        global_scroll.pack(fill="both", expand=True)
        global_scroll.grid_columnconfigure(0, weight=1)

        # ── Fill per-signal tabs ──────────────────────────────────────────────
        for sig_name, res in per_sig_results.items():
            parent    = tabs[sig_name]
            t_centers = res["t_centers"]
            preds     = res["preds"]
            scores    = res["scores"]
            row       = 0

            self._add_section_header_in(
                parent, sig_name, preds, scores, row
            )
            row += 1

            fs = self._fs_map.get(sig_name, 2000)

            if "boxplots" in selected_plots:
                fig = plot_boxplots(res["phase_feats"], sig_name)
                self._add_plot_in(parent, "Feature distributions", fig, row)
                row += 1

            if "roc" in selected_plots:
                fig = plot_roc(res["y_session"], scores, sig_name)
                self._add_plot_in(parent, "ROC curve", fig, row)
                row += 1

            if "timeline" in selected_plots:
                fig = _plot_stress_timeline(
                    t_centers, preds, scores, sig_name,
                    self._stress_intervals,
                )
                self._add_plot_in(parent, "Stress timeline", fig, row)
                row += 1

            if "rr_detail" in selected_plots and sig_name == "ECG":
                fig = _plot_rr_detail(
                    signals[sig_name], fs,
                    self._stress_intervals, self._calming_intervals,
                )
                self._add_plot_in(parent, "RR interval detail", fig, row)
                row += 1

        # ── Fill Global tab ───────────────────────────────────────────────────
        g_row = 0

        self._add_section_label(
            global_scroll,
            "Combined stress probability — all signals",
            g_row,
        )
        g_row += 1

        fig_prob = _plot_global_stress_probability(
            stress_map,
            self._stress_intervals,
            self._calming_intervals,
        )
        self._add_plot_in(global_scroll, "", fig_prob, g_row)
        g_row += 1

        self._add_section_label(
            global_scroll,
            "Mean stress probability heatmap — blue → yellow → red",
            g_row,
        )
        g_row += 1

        fig_heat = _plot_global_heatmap_mean(
            stress_map,
            self._stress_intervals,
            self._calming_intervals,
        )
        self._add_plot_in(global_scroll, "", fig_heat, g_row)

        # Hide control panel
        self._set_status(f"Done — {len(stress_map)} signal(s) analysed.")
        self._ctrl.grid_remove()
        self.grid_columnconfigure(0, weight=0)

        return stress_map