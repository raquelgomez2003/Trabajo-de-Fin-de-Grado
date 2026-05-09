"""
app_window.py
Main application window — orchestrates setup, viewer, and analysis.

Layout
------
  Left panel  : subject selector + phase interval inputs + Run Analysis button
  Right area  : CTkTabview with [Signal Viewer] and [Analysis] tabs
"""

from __future__ import annotations
import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt

from app.core.config import RF_WINDOW_SEC, RF_STEP_SEC
from app.core.data_loader import load_subject_folder, time_axis
from app.core.models import compute_bpm

from app.gui.setup_window import SetupWindow
from app.gui.viewer_window import ViewerWindow
from app.gui.analysis_window import AnalysisWindow

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class AppWindow(ctk.CTk):
    """Root window of BioVisor."""

    def __init__(self):
        super().__init__()
        self.title("BioVisor — Biomedical Signal Viewer & Analyser")
        self.geometry("1400x820")
        self.minsize(1100, 650)

        # Session state (populated by SetupWindow)
        self._cfg: dict        = {}
        self._signals: dict    = {}
        self._stress_map: dict = {}
        self._current_subject: int | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Open setup immediately
        self.after(100, self._open_setup)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Left panel ──────────────────────────────────────────────────────
        self._left = ctk.CTkFrame(self, width=230)
        self._left.grid(row=0, column=0, sticky="ns", padx=(10, 4), pady=10)
        self._left.grid_propagate(False)

        ctk.CTkLabel(self._left, text="BioVisor",
                     font=("Arial", 17, "bold")).pack(pady=(14, 2))
        ctk.CTkLabel(self._left, text="Biomedical Signal Analyser",
                     font=("Arial", 10), text_color="gray").pack(pady=(0, 14))

        ctk.CTkButton(self._left, text="⚙  New Session",
                      command=self._open_setup).pack(fill="x", padx=14, pady=4)

        # Subject buttons (built dynamically after setup)
        self._frame_subjects = ctk.CTkScrollableFrame(self._left, label_text="Subjects", height=180)
        self._frame_subjects.pack(fill="x", padx=10, pady=(10, 6))

        # Phase intervals
        ctk.CTkLabel(self._left, text="Phase Intervals (seconds)",
                     font=("Arial", 11, "bold")).pack(pady=(10, 2))

        def _phase_row(label, color):
            f = ctk.CTkFrame(self._left, fg_color="transparent")
            f.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(f, text=label, width=70,
                         text_color=color).pack(side="left")
            e1 = ctk.CTkEntry(f, width=58, placeholder_text="start")
            e1.pack(side="left", padx=2)
            e2 = ctk.CTkEntry(f, width=58, placeholder_text="end")
            e2.pack(side="left", padx=2)
            return e1, e2

        self._e_calm_s, self._e_calm_e   = _phase_row("Calming", "#2a8a4a")
        self._e_stress_s, self._e_stress_e = _phase_row("Stress",  "#aa2222")
        self._e_relax_s, self._e_relax_e = _phase_row("Relax",   "#2244aa")

        ctk.CTkButton(self._left, text="Update Viewer",
                      command=self._update_viewer).pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkButton(self._left, text="▶  Run Analysis",
                      fg_color="#336699", hover_color="#224477",
                      command=self._run_analysis).pack(fill="x", padx=14, pady=4)
        ctk.CTkButton(self._left, text="Reset View",
                      fg_color="transparent", border_width=1,
                      command=self._reset_view).pack(fill="x", padx=14, pady=(4, 14))

        self._lbl_status = ctk.CTkLabel(self._left, text="No session loaded.",
                                        text_color="gray", wraplength=200, font=("Arial", 10))
        self._lbl_status.pack(padx=10, pady=4)

        # ── Right area (tabs) ────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(self)
        self._tabs.grid(row=0, column=1, sticky="nsew", padx=(4, 10), pady=10)
        self._tabs.add("Signal Viewer")
        self._tabs.add("Analysis")

        self._viewer = ViewerWindow(self._tabs.tab("Signal Viewer"))
        self._viewer.pack(fill="both", expand=True)

        self._analysis = AnalysisWindow(self._tabs.tab("Analysis"))
        self._analysis.pack(fill="both", expand=True)

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _open_setup(self):
        SetupWindow(self, on_confirm=self._on_setup_confirmed)

    def _on_setup_confirmed(self, cfg: dict):
        self._cfg = cfg
        self._signals = {}
        self._stress_map = {}
        self._current_subject = None
        self._rebuild_subject_buttons()
        self._set_status(
            f"Session ready\n"
            f"Device: {cfg['device']}\n"
            f"Subjects: {cfg['n_subjects']}\n"
            f"Signals: {', '.join(cfg['signals'])}"
        )

    def _rebuild_subject_buttons(self):
        for w in self._frame_subjects.winfo_children():
            w.destroy()
        n = self._cfg.get("n_subjects", 0)
        for i in range(1, n + 1):
            ctk.CTkButton(
                self._frame_subjects, text=f"Subject {i}",
                command=lambda n=i: self._load_subject(n),
            ).pack(fill="x", pady=3)

    # ── Subject loading ────────────────────────────────────────────────────────

    def _load_subject(self, num: int):
        cfg = self._cfg
        if not cfg:
            messagebox.showinfo("No session", "Please configure a session first.")
            return

        folder     = cfg["folder"]
        device     = cfg["device"]
        signals_to_load = cfg["signals"]

        # Try Base1_SujetoN / SubjectN subfolder patterns
        candidates = [
            folder,                                              # flat: all in root
            f"{folder}/Base1_Sujeto{num}/Biopac data",          # original structure
            f"{folder}/Subject{num}",
            f"{folder}/S{num:02d}",
        ]
        loaded = {}
        for candidate in candidates:
            try:
                loaded = load_subject_folder(candidate, device, signals_to_load)
                if loaded:
                    break
            except FileNotFoundError:
                continue

        if not loaded:
            messagebox.showerror("Load error",
                                 f"Could not find signal files for Subject {num}.\n"
                                 f"Checked inside: {folder}")
            return

        self._signals         = loaded
        self._current_subject = num
        self._stress_map      = {}

        self._set_status(
            f"Subject {num} loaded\n"
            f"{len(loaded)} signal(s): {', '.join(loaded.keys())}"
        )
        self._update_viewer()

    # ── Viewer update ──────────────────────────────────────────────────────────

    def _update_viewer(self):
        if not self._signals:
            return

        phase_intervals = self._read_phase_intervals()
        bpm_data = None
        if "ECG" in self._signals:
            bpm_data = compute_bpm(
                self._signals["ECG"],
                self._cfg.get("fs", {}).get("ECG", 2000),
            )

        self._viewer.render(
            self._signals,
            self._cfg.get("fs", {}),
            phase_intervals,
            self._stress_map,
            bpm_data,
        )
        self._tabs.set("Signal Viewer")

    def _reset_view(self):
        self._viewer.reset_view()

    # ── Analysis ───────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._signals:
            messagebox.showinfo("No data", "Load a subject before running analysis.")
            return

        phase_intervals = self._read_phase_intervals()
        calming_intervals = phase_intervals.get("calming", [])
        stress_intervals  = phase_intervals.get("vexing", [])

        if not stress_intervals:
            messagebox.showwarning(
                "No stress interval",
                "Enter at least the Stress start/end times before running analysis."
            )
            return

        self._set_status("Running analysis…")
        self.update_idletasks()

        self._stress_map = self._analysis.run_analysis(
            self._signals,
            self._cfg.get("fs", {}),
            stress_intervals,
            calming_intervals,
        )

        self._tabs.set("Analysis")
        self._set_status(
            f"Analysis complete\n"
            f"Signals analysed: {', '.join(self._stress_map.keys())}"
        )

        # Auto-refresh viewer with stress overlay
        if self._stress_map:
            self._update_viewer()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _read_phase_intervals(self) -> dict[str, list[tuple[float, float]]]:
        """Parse the phase entry fields; silently skip invalid/empty pairs."""
        result: dict[str, list[tuple[float, float]]] = {
            "calming": [], "vexing": [], "relax": []
        }

        def _parse(es, ee, key):
            try:
                s = float(es.get())
                e = float(ee.get())
                result[key].append((s, e))
            except ValueError:
                pass

        _parse(self._e_calm_s,   self._e_calm_e,   "calming")
        _parse(self._e_stress_s, self._e_stress_e, "vexing")
        _parse(self._e_relax_s,  self._e_relax_e,  "relax")
        return result

    def _set_status(self, msg: str):
        self._lbl_status.configure(text=msg)

    def _on_close(self):
        plt.close("all")
        self.destroy()
