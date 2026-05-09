"""
analysis_window.py
Analysis tab — stress prediction summary, boxplots per phase, ROC curves.
One tab per signal inside a CTkTabview.
"""

from __future__ import annotations
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app.core.models import (
    StressModel, extract_features_windowed,
    build_labels_from_intervals, build_physiological_labels,
    FEATURE_NAMES, FEATURE_DIM,
)
from app.core.config import RF_WINDOW_SEC, RF_STEP_SEC
from app.core.plotter import plot_boxplots, plot_roc


class AnalysisWindow(ctk.CTkFrame):
    """
    Embedded frame used as a tab inside AppWindow.
    Call run_analysis(signals, fs_map, stress_intervals) to populate.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._figs: list[plt.Figure] = []
        self._build_placeholder()

    def _build_placeholder(self):
        ctk.CTkLabel(
            self, text="Run analysis after loading a subject.",
            text_color="gray", font=("Arial", 13),
        ).pack(expand=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def run_analysis(
        self,
        signals: dict[str, np.ndarray],
        fs_map: dict[str, float],
        stress_intervals: list[tuple[float, float]],
        calming_intervals: list[tuple[float, float]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        1. Derive stress labels from physiology (BPM, EDA, RESP, etc.) via majority vote.
        2. Train one RandomForest per signal using those physiological labels.
        3. Generate boxplots (features split by session phase) and ROC curves
           (physiological predictions vs session-interval ground truth).
        Returns stress_map: {signal_name: (t_centers, predictions)} for the viewer.
        """
        for w in self.winfo_children():
            w.destroy()
        for fig in self._figs:
            plt.close(fig)
        self._figs.clear()

        if not signals:
            self._build_placeholder()
            return {}

        ctk.CTkLabel(self, text="Analysis Results",
                     font=("Arial", 14, "bold")).pack(pady=(10, 4))

        # ── Step 1: physiological labels (cross-signal majority vote) ──────
        t_phys, y_phys = build_physiological_labels(
            signals, fs_map, RF_WINDOW_SEC, RF_STEP_SEC
        )

        tab_view = ctk.CTkTabview(self)
        tab_view.pack(fill="both", expand=True, padx=8, pady=8)

        stress_map = {}

        for sig_name, signal in signals.items():
            fs = fs_map.get(sig_name, 2000)

            # ── Per-signal feature matrix ───────────────────────────────────
            X_all, t_centers = extract_features_windowed(
                signal, fs, sig_name, RF_WINDOW_SEC, RF_STEP_SEC
            )
            if len(X_all) == 0:
                continue

            # ── Align physiological labels to this signal's windows ─────────
            n_win = min(len(X_all), len(t_phys))
            if n_win == 0:
                continue
            X_train = X_all[:n_win]
            y_train = y_phys[:n_win]

            if len(np.unique(y_train)) < 2:
                continue

            # ── Train RF on physiological labels ───────────────────────────
            model  = StressModel()
            model.fit(X_train, y_train)
            preds  = model.predict(X_all)
            scores = model.predict_proba(X_all)
            stress_map[sig_name] = (t_centers, preds)

            # ── Session-interval labels for ROC comparison ──────────────────
            y_session = build_labels_from_intervals(t_centers, stress_intervals)

            # ── Phase features for boxplots ─────────────────────────────────
            phase_feats = _split_by_phase(
                X_all, t_centers, calming_intervals, stress_intervals
            )

            fig_box = plot_boxplots(phase_feats, sig_name)
            fig_roc = plot_roc(y_session, scores, sig_name)
            self._figs.extend([fig_box, fig_roc])

            tab = tab_view.add(sig_name)
            self._populate_signal_tab(tab, fig_box, fig_roc,
                                      model, sig_name,
                                      y_session, scores, preds, y_train)

        if not stress_map:
            ctk.CTkLabel(
                self,
                text="Not enough data to build physiological labels.\n"
                     "Ensure signals are at least 2× the window size "
                     f"({RF_WINDOW_SEC} s).",
                text_color="#cc4400",
            ).pack(pady=10)

        return stress_map

    # ── Tab layout ─────────────────────────────────────────────────────────────

    def _populate_signal_tab(
        self, parent,
        fig_box: plt.Figure,
        fig_roc: plt.Figure,
        model: StressModel,
        sig_name: str,
        y_session: np.ndarray,
        y_scores: np.ndarray,
        preds: np.ndarray,
        y_phys: np.ndarray,
    ):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        # ── Summary bar ────────────────────────────────────────────────────
        summary = ctk.CTkFrame(parent, fg_color="transparent")
        summary.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=4)

        n_stressed  = int(preds.sum())
        n_total     = len(preds)
        # Agreement between physiological detection and session intervals
        n_common    = min(len(preds), len(y_session))
        agreement   = float((preds[:n_common] == y_session[:n_common]).mean()) if n_common else 0.0
        importances = model.feature_importances()
        top_idx     = int(np.argmax(importances))

        ctk.CTkLabel(
            summary,
            text=(f"  {sig_name}  |  "
                  f"Physio-stressed windows: {n_stressed}/{n_total}  |  "
                  f"Agreement with session: {agreement:.1%}  |  "
                  f"Top feature: {FEATURE_NAMES[top_idx]}"),
            font=("Arial", 11),
        ).pack(side="left")

        # ── Boxplots (left) ────────────────────────────────────────────────
        frame_box = ctk.CTkFrame(parent)
        frame_box.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        frame_box.grid_rowconfigure(1, weight=1)
        frame_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame_box, text="Feature Distributions",
                     font=("Arial", 11, "bold")).grid(row=0, column=0, pady=4)
        self._embed_figure(frame_box, fig_box, row=1)

        # ── ROC curve (right) ─────────────────────────────────────────────
        frame_roc = ctk.CTkFrame(parent)
        frame_roc.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        frame_roc.grid_rowconfigure(1, weight=1)
        frame_roc.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame_roc, text="ROC Curve",
                     font=("Arial", 11, "bold")).grid(row=0, column=0, pady=4)
        self._embed_figure(frame_roc, fig_roc, row=1)

    @staticmethod
    def _embed_figure(parent, fig: plt.Figure, row: int = 0):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=row, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(parent, fg_color="transparent", height=28)
        toolbar_frame.grid(row=row + 1, column=0, sticky="ew")
        tb = NavigationToolbar2Tk(canvas, toolbar_frame)
        tb.update()
        tb.pack(side="left")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_by_phase(
    X: np.ndarray,
    t_centers: np.ndarray,
    calming_intervals: list[tuple[float, float]],
    stress_intervals: list[tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Split feature matrix X into calming / stressed / baseline subsets."""
    mask_calm = np.zeros(len(t_centers), dtype=bool)
    for s, e in calming_intervals:
        mask_calm |= (t_centers >= s) & (t_centers <= e)

    mask_stress = np.zeros(len(t_centers), dtype=bool)
    for s, e in stress_intervals:
        mask_stress |= (t_centers >= s) & (t_centers <= e)

    mask_base = ~mask_calm & ~mask_stress

    return {
        "Baseline": X[mask_base]  if mask_base.any()   else np.empty((0, X.shape[1])),
        "Calming":  X[mask_calm]  if mask_calm.any()   else np.empty((0, X.shape[1])),
        "Stress":   X[mask_stress] if mask_stress.any() else np.empty((0, X.shape[1])),
    }