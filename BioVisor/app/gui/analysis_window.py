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


# ── Plot functions ────────────────────────────────────────────────────────────

def _plot_stress_timeline(
    t_centers: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    sig_name: str,
    stress_intervals: list[tuple[float, float]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(t_centers, scores, alpha=0.35, color="#2255aa", label="Stress probability")
    ax.plot(t_centers, scores, lw=0.8, color="#2255aa")
    ax.axhline(0.5, color="#cc0000", lw=0.8, ls="--", label="Decision threshold 0.5")
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.4, label="Session stress interval")
    stressed_t = t_centers[preds == 1]
    if len(stressed_t):
        ax.scatter(stressed_t, np.full(len(stressed_t), 0.02),
                   color="#cc0000", marker="|", s=40, zorder=5, label="Predicted stress")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("P(stress)", fontsize=9)
    ax.set_title(f"Stress probability over time — {sig_name}", fontsize=10, fontweight="bold")
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
        ax.text(0.5, 0.5, "No RR intervals detected — check ECG signal",
                ha="center", va="center")
        ax.set_title("RR interval detail")
        fig.tight_layout()
        return fig

    ax.plot(rr_times, rr_ms, color="#9933aa", lw=0.7, zorder=2)
    ax.scatter(rr_times, rr_ms, color="#9933aa", s=6, zorder=3)

    rr_mean    = np.mean(rr_ms)
    low_thresh = rr_mean - np.std(rr_ms)
    ax.axhline(rr_mean,    color="#777777", lw=1.0, ls="--",
               label=f"Global mean {rr_mean:.0f} ms")
    ax.axhline(low_thresh, color="#cc0000", lw=0.9, ls=":",
               label=f"Stress threshold {low_thresh:.0f} ms")

    mask = rr_ms < low_thresh
    if mask.any():
        ax.scatter(rr_times[mask], rr_ms[mask],
                   color="#cc0000", s=18, zorder=5, label="Short RR (fast HR)")

    for s, e in calming_intervals:
        ax.axvspan(s, e, color="#b6f0c8", alpha=0.35, zorder=0)
        phase_mask = (rr_times >= s) & (rr_times <= e)
        if phase_mask.sum() > 2:
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ms[phase_mask]) ** 2)))
            mean  = float(np.mean(rr_ms[phase_mask]))
            ax.text((s + e) / 2, rr_ms.max() + 10,
                    f"Calming\nmean {mean:.0f} ms\nRMSSD {rmssd:.1f}",
                    ha="center", va="bottom", fontsize=7, color="#1a6b3a",
                    bbox=dict(fc="white", alpha=0.6, ec="none"))

    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.35, zorder=0)
        phase_mask = (rr_times >= s) & (rr_times <= e)
        if phase_mask.sum() > 2:
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ms[phase_mask]) ** 2)))
            mean  = float(np.mean(rr_ms[phase_mask]))
            ax.text((s + e) / 2, rr_ms.max() + 10,
                    f"Stress\nmean {mean:.0f} ms\nRMSSD {rmssd:.1f}",
                    ha="center", va="bottom", fontsize=7, color="#aa2222",
                    bbox=dict(fc="white", alpha=0.6, ec="none"))

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("RR interval (ms)", fontsize=9)
    ax.set_title("Beat-by-beat RR interval — each point = time between two heartbeats\n"
                 "Short RR = fast HR = potential stress",
                 fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def _plot_overview_probability(
    t_centers: np.ndarray,
    scores: np.ndarray,
    stress_intervals: list[tuple[float, float]],
) -> plt.Figure:
    """Gráfica 1 del Overview — curva de probabilidad de estrés."""
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(t_centers, scores, alpha=0.35, color="#2255aa", label="Stress probability")
    ax.plot(t_centers, scores, lw=1.0, color="#2255aa")
    ax.axhline(0.5, color="#cc0000", lw=0.9, ls="--", label="Threshold 0.5")
    for s, e in stress_intervals:
        ax.axvspan(s, e, color="#ffb3b3", alpha=0.35, label="Stress interval")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(stress)", fontsize=9)
    ax.set_title("Stress probability over time", fontsize=10, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_xticklabels([])  # el eje X lo muestra el heatmap de abajo
    fig.tight_layout()
    return fig


def _plot_overview_heatmap(
    t_centers: np.ndarray,
    scores: np.ndarray,
) -> plt.Figure:
    """Gráfica 2 del Overview — heatmap horizontal de la misma probabilidad."""
    fig, ax = plt.subplots(figsize=(11, 1.2))

    # Convertir scores a imagen 2D de 1 fila para imshow
    heatmap = scores.reshape(1, -1)

    im = ax.imshow(
        heatmap,
        aspect="auto",
        cmap="RdYlGn_r",        # verde=bajo estrés, rojo=alto estrés
        vmin=0, vmax=1,
        extent=[t_centers[0], t_centers[-1], 0, 1],
    )

    # Colorbar como leyenda
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.55, fraction=0.15, aspect=40)
    cbar.set_label("P(stress)", fontsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0 (no stress)", "0.25", "0.5", "0.75", "1 (stress)"],
                        fontsize=7)

    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("Stress heatmap", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Main analysis window ──────────────────────────────────────────────────────

class AnalysisWindow(ctk.CTkFrame):

    PLOT_TYPES = [
        ("Boxplots",        "boxplots"),
        ("ROC curve",       "roc"),
        ("Stress timeline", "timeline"),
        ("RR detail",       "rr_detail"),
    ]

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
        self._ctrl.grid(row=0, column=0, sticky="ns", padx=(6, 2), pady=6)
        ctk.CTkLabel(self._ctrl, text="Analysis controls",
                     font=("Arial", 13, "bold")).pack(anchor="w", pady=(8, 4))
        self._build_controls()

        self._results_frame = ctk.CTkFrame(self)
        self._results_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 6), pady=6)
        self._results_frame.grid_columnconfigure(0, weight=1)
        self._results_frame.grid_rowconfigure(0, weight=1)
        ctk.CTkLabel(self._results_frame,
                     text="Configure the controls on the left and press Run.",
                     text_color="gray", font=("Arial", 13)).pack(pady=40)

    def _build_controls(self):
        ctrl = self._ctrl

        self._sig_section = ctk.CTkFrame(ctrl, fg_color="transparent")
        self._sig_section.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(self._sig_section, text="Signals to analyse",
                     font=("Arial", 11, "bold")).pack(anchor="w")
        ctk.CTkLabel(self._sig_section, text="(load a subject first)",
                     text_color="gray", font=("Arial", 10)).pack(anchor="w")

        ctk.CTkLabel(ctrl, text="Plot types",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(8, 2))
        self._plot_vars = {}
        for label, key in self.PLOT_TYPES:
            var = ctk.BooleanVar(value=True)
            self._plot_vars[key] = var
            ctk.CTkCheckBox(ctrl, text=label, variable=var,
                            font=("Arial", 11)).pack(anchor="w", padx=4, pady=1)

        ctk.CTkLabel(ctrl, text="Window size (s)",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 2))
        self._window_var = ctk.StringVar(value=str(RF_WINDOW_SEC))
        ctk.CTkEntry(ctrl, textvariable=self._window_var, height=28).pack(fill="x", padx=4)

        ctk.CTkLabel(ctrl, text="Step size (s)",
                     font=("Arial", 11, "bold")).pack(anchor="w", pady=(6, 2))
        self._step_var = ctk.StringVar(value=str(RF_STEP_SEC))
        ctk.CTkEntry(ctrl, textvariable=self._step_var, height=28).pack(fill="x", padx=4)

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

    # ── Control helpers ───────────────────────────────────────────────────────

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

    def _get_window_step(self) -> tuple[float, float]:
        try:
            win  = float(self._window_var.get())
            step = float(self._step_var.get())
            assert win > 0 and step > 0 and step <= win
            return win, step
        except Exception:
            return RF_WINDOW_SEC, RF_STEP_SEC

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self._clear_results()

        selected_sigs  = self._selected_signals()
        selected_plots = self._selected_plots()
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

        signals = {s: self._signals[s] for s in selected_sigs if s in self._signals}

        t_phys, y_phys = build_physiological_labels(
            signals, self._fs_map, window_sec, step_sec
        )

        stress_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Tab view con Overview primero + una pestaña por señal
        tab_view = ctk.CTkTabview(self._results_frame)
        tab_view.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self._results_frame.grid_rowconfigure(0, weight=1)

        # Pestaña Overview
        tab_view.add("Overview")
        overview_scroll = ctk.CTkScrollableFrame(tab_view.tab("Overview"))
        overview_scroll.pack(fill="both", expand=True)
        overview_scroll.grid_columnconfigure(0, weight=1)
        overview_row = 0

        # Pestañas por señal
        tabs: dict[str, ctk.CTkScrollableFrame] = {}
        for sig_name in selected_sigs:
            if sig_name in signals:
                tab_view.add(sig_name)
                scroll = ctk.CTkScrollableFrame(tab_view.tab(sig_name))
                scroll.pack(fill="both", expand=True)
                scroll.grid_columnconfigure(0, weight=1)
                tabs[sig_name] = scroll

        row_per_tab: dict[str, int] = {s: 0 for s in tabs}

        # Acumulador de scores para el Overview
        all_t:      np.ndarray | None = None
        all_scores: list[np.ndarray]  = []

        for sig_name in selected_sigs:
            if sig_name not in signals:
                continue

            fs = self._fs_map.get(sig_name, 2000)
            X_all, t_centers = extract_features_windowed(
                signals[sig_name], fs, sig_name, window_sec, step_sec
            )
            if len(X_all) == 0:
                continue

            # Filtro IQR
            Q1  = np.percentile(X_all, 25, axis=0)
            Q3  = np.percentile(X_all, 75, axis=0)
            IQR = Q3 - Q1
            mask_ok   = np.all((X_all >= Q1 - 3 * IQR) & (X_all <= Q3 + 3 * IQR), axis=1)
            X_all     = X_all[mask_ok]
            t_centers = t_centers[mask_ok]
            print(f"[IQR] {sig_name}: {mask_ok.sum()}/{len(mask_ok)} ventanas conservadas")
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
            stress_map[sig_name] = (t_centers, preds)

            # Acumular para Overview
            if all_t is None:
                all_t = t_centers
            all_scores.append(scores[:len(all_t)])

            y_session   = build_labels_from_intervals(t_centers, self._stress_intervals)
            phase_feats = _split_by_phase(
                X_all, t_centers,
                self._calming_intervals, self._stress_intervals
            )

            parent = tabs[sig_name]
            row    = row_per_tab[sig_name]

            self._add_section_header_in(parent, sig_name, preds, scores, row)
            row += 1

            if "boxplots" in selected_plots:
                fig = plot_boxplots(phase_feats, sig_name)
                self._add_plot_in(parent, "Feature distributions", fig, row)
                row += 1

            if "roc" in selected_plots:
                fig = plot_roc(y_session, scores, sig_name)
                self._add_plot_in(parent, "ROC curve", fig, row)
                row += 1

            if "timeline" in selected_plots:
                fig = _plot_stress_timeline(
                    t_centers, preds, scores, sig_name, self._stress_intervals
                )
                self._add_plot_in(parent, "Stress timeline", fig, row)
                row += 1

            if "rr_detail" in selected_plots and sig_name == "ECG":
                fig = _plot_rr_detail(
                    signals[sig_name], fs,
                    self._stress_intervals, self._calming_intervals
                )
                self._add_plot_in(parent, "Beat-by-beat RR interval", fig, row)
                row += 1

            row_per_tab[sig_name] = row

        # ── Overview: probabilidad media + heatmap ────────────────────────
        if all_t is not None and all_scores:
            n_win        = min(len(s) for s in all_scores)
            mean_scores  = np.mean([s[:n_win] for s in all_scores], axis=0)
            t_overview   = all_t[:n_win]

            fig_prob = _plot_overview_probability(
                t_overview, mean_scores, self._stress_intervals
            )
            self._add_plot_in(overview_scroll,
                              "Stress probability — mean across all signals",
                              fig_prob, overview_row)
            overview_row += 1

            fig_heat = _plot_overview_heatmap(t_overview, mean_scores)
            self._add_plot_in(overview_scroll,
                              "Stress heatmap", fig_heat, overview_row)
            overview_row += 1

        if not stress_map:
            ctk.CTkLabel(
                self._results_frame,
                text=f"Not enough data.\nSignals need ≥ 2× window size ({window_sec:.0f} s).",
                text_color="#cc4400", font=("Arial", 12),
            ).grid(row=0, column=0, pady=20)
            self._set_status("No results — check window size.")
        else:
            self._set_status(f"Done — {len(stress_map)} signal(s) analysed.")

        return stress_map

    # ── Results helpers ───────────────────────────────────────────────────────

    def _clear_results(self):
        for w in self._results_frame.winfo_children():
            w.destroy()
        for fig in self._figs:
            plt.close(fig)
        self._figs.clear()
        self._results_frame.grid_rowconfigure(0, weight=1)

    def _add_section_header_in(self, parent, sig_name, preds, scores, row):
        n_stressed = int(preds.sum())
        n_total    = len(preds)
        frame = ctk.CTkFrame(parent, fg_color="#336699", corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=6, pady=(8, 2))
        ctk.CTkLabel(
            frame,
            text=(f"  Stressed windows: {n_stressed} / {n_total}   ·   "
                  f"Mean stress prob: {scores.mean():.2f}"),
            font=("Arial", 11, "bold"),
            text_color="white",
        ).pack(side="left", padx=8, pady=6)

    def _add_plot_in(self, parent, title: str, fig: plt.Figure, row: int):
        self._figs.append(fig)
        card = ctk.CTkFrame(parent, corner_radius=8)
        card.grid(row=row, column=0, sticky="ew", padx=6, pady=4)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text=title,
                     font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="ew", padx=6)

        tb_frame = ctk.CTkFrame(card, fg_color="transparent", height=26)
        tb_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        tb = NavigationToolbar2Tk(canvas, tb_frame)
        tb.update()
        tb.pack(side="left")

    def _set_status(self, msg: str):
        if self._status_var:
            self._status_var.set(msg)


# ── Helper ────────────────────────────────────────────────────────────────────

def _split_by_phase(
    X: np.ndarray,
    t_centers: np.ndarray,
    calming_intervals: list[tuple[float, float]],
    stress_intervals: list[tuple[float, float]],
) -> dict[str, np.ndarray]:
    mask_calm   = np.zeros(len(t_centers), dtype=bool)
    mask_stress = np.zeros(len(t_centers), dtype=bool)

    for s, e in calming_intervals:
        mask_calm   |= (t_centers >= s) & (t_centers <= e)
    for s, e in stress_intervals:
        mask_stress |= (t_centers >= s) & (t_centers <= e)

    mask_base = ~mask_calm & ~mask_stress
    ncols     = X.shape[1] if X.ndim == 2 else 1

    return {
        "Baseline": X[mask_base]   if mask_base.any()   else np.empty((0, ncols)),
        "Calming":  X[mask_calm]   if mask_calm.any()   else np.empty((0, ncols)),
        "Stress":   X[mask_stress] if mask_stress.any() else np.empty((0, ncols)),
    }