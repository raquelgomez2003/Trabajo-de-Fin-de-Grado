"""
viewer_window.py
Signal Viewer tab — shows all loaded signals with phase shading,
predicted-stress markers, and ECG BPM overlay.
"""

from __future__ import annotations
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from app.core.plotter import plot_signals_with_stress


class ViewerWindow(ctk.CTkFrame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas  = None
        self._toolbar = None
        self._fig: plt.Figure | None = None
        self._build()

    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._frame_toolbar = ctk.CTkFrame(self, height=36, fg_color="transparent")
        self._frame_toolbar.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))

        self._frame_plot = ctk.CTkFrame(self)
        self._frame_plot.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._frame_plot.grid_rowconfigure(0, weight=1)
        self._frame_plot.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self._frame_plot, text="Load a subject to display signals.",
                     text_color="gray", font=("Arial", 13)).grid(row=0, column=0)

    def render(self, signals: dict, fs_map: dict, phase_intervals: dict,
               stress_map: dict, bpm_data=None, rr_data=None):
        """Draw (or redraw) the viewer figure."""
        self._clear()
        self._fig = plot_signals_with_stress(
            signals, fs_map, phase_intervals, stress_map, bpm_data, rr_data
        )
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._frame_plot)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._toolbar = NavigationToolbar2Tk(self._canvas, self._frame_toolbar)
        self._toolbar.update()
        self._toolbar.pack(side="left")

    def reset_view(self):
        if self._toolbar:
            self._toolbar.home()

    def _clear(self):
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