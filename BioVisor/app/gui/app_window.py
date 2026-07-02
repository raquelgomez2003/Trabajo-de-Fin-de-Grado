"""
app_window.py
Main application window — orchestrates setup, viewer, and analysis.
"""

from __future__ import annotations
import os
import sys
import subprocess
import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt

from app.core.data_loader import load_subject_folder
from app.gui.setup_window import SetupWindow
from app.gui.viewer_window import ViewerWindow
from app.gui.analysis_window import AnalysisWindow

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class AppWindow(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("BioVisor — Biomedical Signal Viewer & Analyser")
        self.geometry("1400x820")
        self.minsize(1100, 650)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ico = os.path.join(BASE_DIR, "assets", "logo.ico")
        if os.path.exists(ico):
            self.iconbitmap(ico)

        self._cfg:             dict       = {}
        self._signals:         dict       = {}
        self._stress_map:      dict       = {}
        self._current_subject: int | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._open_setup)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._left = ctk.CTkFrame(self, width=230)
        self._left.grid(row=0, column=0, sticky="ns", padx=(10, 4), pady=10)
        self._left.grid_propagate(False)

        ctk.CTkLabel(self._left, text="BioVisor",
                     font=("Arial", 17, "bold")).pack(pady=(14, 2))
        ctk.CTkLabel(self._left, text="Biomedical Signal Analyser",
                     font=("Arial", 10), text_color="gray").pack(pady=(0, 14))

        ctk.CTkButton(self._left, text="⚙  New Session",
                      command=self._open_setup).pack(fill="x", padx=14, pady=4)

        self._filter_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self._left, text="Apply artifact filter",
                        variable=self._filter_var,
                        font=("Arial", 11)).pack(padx=14, pady=(4, 0), anchor="w")

        self._frame_subjects = ctk.CTkScrollableFrame(
            self._left, label_text="Subjects", height=180)
        self._frame_subjects.pack(fill="x", padx=10, pady=(10, 6))

        ctk.CTkLabel(self._left, text="Phase Intervals (seconds)",
                     font=("Arial", 11, "bold")).pack(pady=(10, 2))

        self._e_calm_s,   self._e_calm_e   = self._phase_row("Calming", "#2a8a4a")
        self._e_stress_s, self._e_stress_e = self._phase_row("Stress",  "#aa2222")

        ctk.CTkLabel(self._left, text="Relax = everything outside\nCalming & Stress",
                     font=("Arial", 9), text_color="gray").pack(pady=(0, 4))

        ctk.CTkButton(self._left, text="Update Viewer",
                      command=self._update_viewer).pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkButton(self._left, text="▶  Run Analysis",
                      fg_color="#336699", hover_color="#224477",
                      command=self._run_analysis).pack(fill="x", padx=14, pady=4)

        ctk.CTkButton(self._left, text="↺  Reset App",
                      fg_color="transparent", border_width=1,
                      text_color="#aa2222", border_color="#aa2222",
                      hover_color="#ffeeee",
                      command=self._reset_app).pack(fill="x", padx=14, pady=(4, 14))

        self._lbl_status = ctk.CTkLabel(self._left, text="No session loaded.",
                                        text_color="gray", wraplength=200,
                                        font=("Arial", 10))
        self._lbl_status.pack(padx=10, pady=4)

        self._tabs = ctk.CTkTabview(self)
        self._tabs.grid(row=0, column=1, sticky="nsew", padx=(4, 10), pady=10)
        self._tabs.add("Signal Viewer")
        self._tabs.add("Analysis")

        self._viewer = ViewerWindow(self._tabs.tab("Signal Viewer"))
        self._viewer.pack(fill="both", expand=True)

        self._analysis = AnalysisWindow(self._tabs.tab("Analysis"))
        self._analysis.pack(fill="both", expand=True)

    def _phase_row(self, label: str, color: str) -> tuple:
        f = ctk.CTkFrame(self._left, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(f, text=label, width=70, text_color=color).pack(side="left")
        e1 = ctk.CTkEntry(f, width=58, placeholder_text="start")
        e1.pack(side="left", padx=2)
        e2 = ctk.CTkEntry(f, width=58, placeholder_text="end")
        e2.pack(side="left", padx=2)
        return e1, e2

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _open_setup(self):
        SetupWindow(self, on_confirm=self._on_setup_confirmed)

    def _on_setup_confirmed(self, cfg: dict):
        self._cfg             = cfg
        self._signals         = {}
        self._stress_map      = {}
        self._current_subject = None
        self._rebuild_subject_buttons()
        self._set_status(
            f"Session ready\nDevice: {cfg['device']}\n"
            f"Subjects: {cfg['n_subjects']}\n"
            f"Signals: {', '.join(cfg['signals'])}"
        )

    def _rebuild_subject_buttons(self):
        for w in self._frame_subjects.winfo_children():
            w.destroy()
        for i in range(1, self._cfg.get("n_subjects", 0) + 1):
            ctk.CTkButton(
                self._frame_subjects, text=f"Subject {i}",
                command=lambda n=i: self._load_subject(n),
            ).pack(fill="x", pady=3)

    # ── Subject loading ───────────────────────────────────────────────────────

    def _load_subject(self, num: int):
        if not self._cfg:
            messagebox.showinfo("No session", "Please configure a session first.")
            return

        folder = self._cfg["folder"]
        device = self._cfg["device"]
        fs_map = dict(self._cfg.get("fs", {}))

        popup = ctk.CTkToplevel(self)
        popup.title(f"Loading Subject {num}")
        popup.geometry("500x320")
        popup.resizable(False, False)
        popup.grab_set()

        ctk.CTkLabel(popup, text=f"Loading Subject {num}…",
                     font=("Arial", 13, "bold")).pack(pady=(16, 6))

        log_box = ctk.CTkTextbox(popup, width=460, height=210, font=("Courier", 10))
        log_box.pack(padx=16, pady=(0, 16))
        log_box.configure(state="disabled")

        def _log(msg: str):
            log_box.configure(state="normal")
            log_box.insert("end", msg + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")
            popup.update()

        class _LogRedirect:
            def write(self, msg):
                if msg.strip():
                    _log(msg.strip())
            def flush(self):
                pass

        old_stdout = sys.stdout
        sys.stdout = _LogRedirect()

        if device == "Biopac":
            candidates = [
                os.path.join(folder, f"Base1_Sujeto{num}", "Biopac data"),
                os.path.join(folder, f"Base1_Sujeto{num}"),
                os.path.join(folder, f"Subject{num}"),
                os.path.join(folder, f"S{num:02d}"),
                folder,
            ]
        else:  # Empatica
            candidates = [
                os.path.join(folder, f"Base1_Sujeto{num}", "Empatica_data"),
                os.path.join(folder, f"Base1_Sujeto{num}", "Empatica data"),
                os.path.join(folder, f"Base1_Sujeto{num}", "Empatica"),
                os.path.join(folder, f"Subject{num}"),
                os.path.join(folder, f"S{num:02d}"),
                folder,
            ]

        loaded = {}
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            try:
                loaded = load_subject_folder(
                    candidate, device, self._cfg["signals"],
                    fs_map=fs_map,
                    apply_filters=self._filter_var.get(),
                )
                if loaded:
                    self._cfg["fs"] = fs_map
                    break
            except Exception as e:
                print(f"[SKIP] {candidate}: {e}")
                continue

        sys.stdout = old_stdout

        if not loaded:
            popup.destroy()
            messagebox.showerror("Load error",
                                 f"Could not find signal files for Subject {num}.\n"
                                 f"Checked inside: {folder}")
            return

        _log(f"\n✓ Done — {len(loaded)} signal(s) loaded.")
        ctk.CTkButton(popup, text="Continue →",
                      command=popup.destroy).pack(pady=(0, 12))

        self._signals         = loaded
        self._current_subject = num
        self._stress_map      = {}
        self._set_status(f"Subject {num} loaded\n"
                         f"{len(loaded)} signal(s): {', '.join(loaded.keys())}")

        popup.wait_window()

        # ── Session times + Empatica alignment ───────────────────────────────
        if self._cfg.get("device") == "Empatica":
            self._load_empatica_session(num)
        else:
            self._try_load_session_times(num)

        self._update_viewer()

    # ── Session times ─────────────────────────────────────────────────────────

    def _load_empatica_session(self, num: int) -> None:
        """
        For Empatica sessions:
        - If Subject*_Session_Times.csv exists inside Empatica_data:
            load times directly (no offset needed).
        - Otherwise:
            compute offset via cross-correlation (align_hr_ecg.py),
            clip all loaded signals to the ECG-matching window,
            and fill session times from the subject-level CSV if present.
        """
        folder       = self._cfg.get("folder", "")
        subject_dir  = os.path.join(folder, f"Base1_Sujeto{num}")
        empatica_dir = os.path.join(subject_dir, "Empatica_data")

        # ── Check for Session_Times inside Empatica_data ──────────────────────
        session_csv = None
        if os.path.isdir(empatica_dir):
            for entry in os.scandir(empatica_dir):
                if entry.is_file() and "session_times" in entry.name.lower():
                    session_csv = entry.path
                    break

        if session_csv is not None:
            # Session times available → load directly, no offset needed
            self._fill_phase_entries_from_csv(session_csv)
            print(f"[OK] Session times from Empatica folder: {os.path.basename(session_csv)}")
            return

        # ── No session times in Empatica folder → compute offset ─────────────
        biopac_dir = os.path.join(subject_dir, "Biopac data")
        ecg_path, hr_path = None, None

        if os.path.isdir(biopac_dir):
            for entry in os.scandir(biopac_dir):
                if "ECG" in entry.name.upper() and entry.name.lower().endswith(".csv"):
                    ecg_path = entry.path
                    break

        if os.path.isdir(empatica_dir):
            for entry in os.scandir(empatica_dir):
                if "HR" in entry.name.upper() and entry.name.lower().endswith(".csv"):
                    hr_path = entry.path
                    break

        if ecg_path is None or hr_path is None:
            print("[WARN] Cannot compute offset — ECG or HR not found")
            # Still try to load session times from subject root
            self._try_load_session_times(num)
            return

        # Run align_hr_ecg.py
        script_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "core", "align_hr_ecg.py"
        ))

        offset = 0
        if os.path.exists(script_path):
            try:
                result = subprocess.run(
                    [sys.executable, script_path, ecg_path, hr_path],
                    capture_output=True, text=True, timeout=120
                )
                offset = int(result.stdout.strip().splitlines()[-1].strip())
                print(f"[OK] Empatica HR offset for Subject {num}: {offset} s")
            except Exception as e:
                print(f"[WARN] Could not compute HR offset: {e}")
        else:
            print(f"[WARN] align_hr_ecg.py not found at {script_path}")

        # Load session times from subject root and add offset so phases align
        self._try_load_session_times(num)
        if offset > 0:
            for entry_start, entry_end in [
                (self._e_calm_s,   self._e_calm_e),
                (self._e_stress_s, self._e_stress_e),
            ]:
                try:
                    s = int(entry_start.get())
                    e = int(entry_end.get())
                    entry_start.delete(0, "end")
                    entry_start.insert(0, str(s + offset))
                    entry_end.delete(0, "end")
                    entry_end.insert(0, str(e + offset))
                except ValueError:
                    pass

        self._set_status(
            f"Subject {num} loaded\n"
            f"{len(self._signals)} signal(s)\n"
            f"HR offset: +{offset}s"
        )

    def _fill_phase_entries_from_csv(self, csv_path: str) -> None:
        """Fill calming/stress entries from a semicolon-separated session times CSV."""
        try:
            import csv as _csv
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = _csv.DictReader(f, delimiter=";")
                for row in reader:
                    cs = row.get("calming_start", "").strip().replace(".", "")
                    ce = row.get("calming_end",   "").strip().replace(".", "")
                    vs = row.get("vexing_start",  "").strip().replace(".", "")
                    ve = row.get("vexing_end",    "").strip().replace(".", "")
                    if not cs:
                        continue
                    for entry_w, val in [
                        (self._e_calm_s,   cs),
                        (self._e_calm_e,   ce),
                        (self._e_stress_s, vs),
                        (self._e_stress_e, ve),
                    ]:
                        entry_w.delete(0, "end")
                        entry_w.insert(0, val)
                    break
        except Exception as e:
            print(f"[WARN] Could not read {csv_path}: {e}")

    def _try_load_session_times(self, num: int) -> None:
        """Load session times from subject root folder (Biopac case)."""
        folder      = self._cfg.get("folder", "")
        subject_dir = os.path.join(folder, f"Base1_Sujeto{num}")
        if not os.path.isdir(subject_dir):
            return
        for entry in os.scandir(subject_dir):
            if entry.is_file() and "session_times" in entry.name.lower():
                self._fill_phase_entries_from_csv(entry.path)
                print(f"[OK] Session times: {entry.name}")
                return

    # ── Viewer ────────────────────────────────────────────────────────────────

    def _update_viewer(self):
        if not self._signals:
            return
        phase_intervals = self._read_phase_intervals()
        self._viewer.render(
            self._signals,
            self._cfg.get("fs", {}),
            phase_intervals,
            self._stress_map,
            bpm_data=None,
            rr_data=None,
        )
        self._tabs.set("Signal Viewer")

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._signals:
            messagebox.showinfo("No data", "Load a subject before running analysis.")
            return
        phase_intervals = self._read_phase_intervals()
        self._analysis.load_data(
            self._signals,
            self._cfg.get("fs", {}),
            phase_intervals.get("vexing",  []),
            phase_intervals.get("calming", []),
        )
        self._tabs.set("Analysis")
        self._set_status("Data loaded into Analysis tab.\n"
                         "Select plots and press ▶ Run analysis.")

    # ── Reset App ─────────────────────────────────────────────────────────────

    def _reset_app(self):
        self._signals         = {}
        self._stress_map      = {}
        self._current_subject = None

        self._viewer._clear()
        ctk.CTkLabel(
            self._viewer._frame_plot,
            text="Load a subject to display signals.",
            text_color="gray", font=("Arial", 13),
        ).grid(row=0, column=0)

        self._analysis.reset_for_new_subject()

        for entry in (self._e_calm_s, self._e_calm_e,
                      self._e_stress_s, self._e_stress_e):
            entry.delete(0, "end")

        self._tabs.set("Signal Viewer")

        if self._cfg:
            self._set_status(
                f"App reset.\nSession: {self._cfg.get('device', '')}\n"
                f"Select a subject to continue."
            )
        else:
            self._set_status("App reset.\nNo session loaded.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_phase_intervals(self) -> dict[str, list[tuple[float, float]]]:
        calming, vexing = [], []

        def _parse(es, ee, target):
            try:
                s, e = float(es.get()), float(ee.get())
                if e > s:
                    target.append((s, e))
            except ValueError:
                pass

        _parse(self._e_calm_s,   self._e_calm_e,   calming)
        _parse(self._e_stress_s, self._e_stress_e, vexing)

        relax = []
        if self._signals:
            fs_map  = self._cfg.get("fs", {})
            sig_dur = max(len(sig) / fs_map.get(name, 2000)
                          for name, sig in self._signals.items())
            cursor  = 0.0
            for start, end in sorted(calming + vexing, key=lambda x: x[0]):
                if start > cursor:
                    relax.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < sig_dur:
                relax.append((cursor, sig_dur))

        return {"calming": calming, "vexing": vexing, "relax": relax}

    def _set_status(self, msg: str):
        self._lbl_status.configure(text=msg)

    def _on_close(self):
        plt.close("all")
        self.destroy()
        os._exit(0)