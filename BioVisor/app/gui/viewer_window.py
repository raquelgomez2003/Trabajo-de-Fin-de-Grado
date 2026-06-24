"""
viewer_window.py
Signal Viewer tab — shows all loaded signals with phase shading,
predicted-stress markers, and ECG BPM overlay.

Zoom behaviour
--------------
All subplots share the x-axis (sharex=True in plotter).
When the user zooms or pans, the y-limits of every axes are
automatically adjusted so the visible signal fills the full
panel height (no empty space above/below).
"""

from __future__ import annotations
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from app.core.plotter import plot_signals_with_stress


class ViewerWindow(ctk.CTkFrame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas:  FigureCanvasTkAgg | None  = None
        self._toolbar: NavigationToolbar2Tk | None = None
        self._fig:     plt.Figure | None          = None
        self._axs:     list[plt.Axes]             = []
        self._cids:    list[int]                  = []
        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._frame_toolbar = ctk.CTkFrame(
            self, height=36, fg_color="transparent"
        )
        self._frame_toolbar.grid(
            row=0, column=0, sticky="ew", padx=4, pady=(4, 0)
        )

        self._frame_plot = ctk.CTkFrame(self)
        self._frame_plot.grid(
            row=1, column=0, sticky="nsew", padx=4, pady=4
        )
        self._frame_plot.grid_rowconfigure(0, weight=1)
        self._frame_plot.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self._frame_plot,
            text="Load a subject to display signals.",
            text_color="gray", font=("Arial", 13),
        ).grid(row=0, column=0)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        signals: dict,
        fs_map: dict,
        phase_intervals: dict,
        stress_map: dict,
        bpm_data=None,
        rr_data=None,
    ):
        """Draw (or redraw) the viewer figure."""
        self._clear()

        self._fig = plot_signals_with_stress(
            signals, fs_map, phase_intervals,
            stress_map, bpm_data, rr_data,
        )

        self._axs = self._fig.get_axes()
        self._connect_zoom_callbacks()

        self._canvas = FigureCanvasTkAgg(
            self._fig, master=self._frame_plot
        )
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(
            row=0, column=0, sticky="nsew"
        )

        self._toolbar = NavigationToolbar2Tk(
            self._canvas, self._frame_toolbar
        )
        self._toolbar.update()
        self._toolbar.pack(side="left")

    def reset_view(self):
        if self._toolbar:
            self._toolbar.home()

    # ── Zoom / auto-ylim ──────────────────────────────────────────────────────

    def _connect_zoom_callbacks(self):
        """
        For each axes, listen to xlim_changed.
        When x-range changes, refit the y-limits of that axes to the
        data that is currently visible — so the signal always fills
        the full panel height with no empty space.
        """
        if not self._axs:
            return

        # Pre-compute (t, y) arrays per axes so the callback is fast
        ax_data: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for ax in self._axs:
            pairs = []
            for line in ax.get_lines():
                xd = np.asarray(line.get_xdata(), dtype=float)
                yd = np.asarray(line.get_ydata(), dtype=float)
                if len(xd) and len(yd) and len(xd) == len(yd):
                    pairs.append((xd, yd))
            ax_data[id(ax)] = pairs

        def _on_xlim_changed(ax):
            x0, x1 = ax.get_xlim()
            pairs   = ax_data.get(id(ax), [])
            if not pairs:
                return

            # Collect all y values inside the current x window
            y_visible = []
            for xd, yd in pairs:
                mask = (xd >= x0) & (xd <= x1)
                if mask.any():
                    y_visible.append(yd[mask])

            if not y_visible:
                return

            y_all = np.concatenate(y_visible)
            y_all = y_all[np.isfinite(y_all)]
            if len(y_all) == 0:
                return

            y_lo = float(np.nanmin(y_all))
            y_hi = float(np.nanmax(y_all))
            margin = (y_hi - y_lo) * 0.08 if y_hi != y_lo else 0.5

            ax.set_ylim(y_lo - margin, y_hi + margin)
            self._canvas.draw_idle()

        # Connect to every axes individually (sharex means x is already
        # synchronised; we handle y per-panel)
        for ax in self._axs:
            cid = ax.callbacks.connect("xlim_changed", _on_xlim_changed)
            self._cids.append(cid)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clear(self):
        # Disconnect zoom callbacks
        for ax, cid in zip(self._axs, self._cids):
            try:
                ax.callbacks.disconnect(cid)
            except Exception:
                pass
        self._cids = []
        self._axs  = []

        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar:
            self._toolbar.destroy()
            self._toolbar = None
        if self._fig:
            plt.close(self._fig)
            self._fig = None

        for w in self._frame_plot.winfo_children():
            w.destroy()