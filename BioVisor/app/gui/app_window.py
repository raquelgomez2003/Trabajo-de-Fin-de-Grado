"""
app_window.py
Main application window — orchestrates setup, viewer, and analysis.

Cambios
-------
- Empatica: los tiempos de fase (calming/vexing) se leen DIRECTAMENTE del
  fichero session_times. Se ha eliminado la alineacion por correlacion cruzada
  (align_hr_ecg.py) y el offset ya no se calcula ni se suma.
- Reset App: deja la app como recien arrancada conservando la sesion, restaura
  las frecuencias originales y reconstruye los botones de sujeto.
- Carga: las frecuencias de cada sujeto se siembran siempre desde las
  originales del setup (self._fs_original), evitando corrupcion entre cargas.
"""

from __future__ import annotations
import os
import sys
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
        self._fs_original:     dict       = {}   # fs del setup, NUNCA se muta

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

    # ── Folder helpers ─────────────────────────────────────────────────────────

    def _base_name(self) -> str:
        """
        Nombre de la base derivado de la carpeta seleccionada en el setup.
        Si seleccionas '...\\Base2' -> 'Base2', si seleccionas '...\\Base1' -> 'Base1'.
        Así las rutas de sujeto se construyen como '{base}_Sujeto{num}' para
        cualquier base, sin quedar fijadas a 'Base1'.
        """
        folder = self._cfg.get("folder", "")
        return os.path.basename(os.path.normpath(folder)) if folder else ""

    def _subject_dir(self, num: int) -> str:
        """Carpeta raíz del sujeto: '{folder}\\{base}_Sujeto{num}'."""
        folder = self._cfg.get("folder", "")
        return os.path.join(folder, f"{self._base_name()}_Sujeto{num}")

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _open_setup(self):
        SetupWindow(self, on_confirm=self._on_setup_confirmed)

    def _on_setup_confirmed(self, cfg: dict):
        self._cfg             = cfg
        self._fs_original     = dict(cfg.get("fs", {}))   # copia intacta
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
        # Sembrar SIEMPRE desde las fs originales del setup (evita corrupcion
        # de frecuencias entre cargas sucesivas).
        fs_map = dict(self._fs_original) if self._fs_original else dict(self._cfg.get("fs", {}))
        base   = self._base_name()          # nombre de base dinámico (Base1, Base2, …)

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
                os.path.join(folder, f"{base}_Sujeto{num}", "Biopac data"),
                os.path.join(folder, f"{base}_Sujeto{num}"),
                os.path.join(folder, f"Subject{num}"),
                os.path.join(folder, f"S{num:02d}"),
                folder,
            ]
        else:  # Empatica
            # Se apunta DIRECTAMENTE a la subcarpeta Empatica, para que los
            # ficheros crudos Left_/Right_ que hay en la carpeta del sujeto
            # NO tapen a tus Sujeto{num}_*.csv combinados.
            candidates = [
                os.path.join(folder, f"{base}_Sujeto{num}", "Empatica_data"),
                os.path.join(folder, f"{base}_Sujeto{num}", "Empatica data"),
                os.path.join(folder, f"{base}_Sujeto{num}", "Empatica"),
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

        # ── Session times (sin alineacion por correlacion cruzada) ───────────
        if self._cfg.get("device") == "Empatica":
            self._load_empatica_session(num)
        else:
            self._try_load_session_times(num)

        self._update_viewer()

    # ── Session times ─────────────────────────────────────────────────────────

    def _load_empatica_session(self, num: int) -> None:
        """
        Sesiones Empatica: los tiempos de fase (calming/vexing) se leen
        DIRECTAMENTE del fichero session_times. No se calcula ningun offset por
        correlacion cruzada. Se busca primero dentro de Empatica_data y, si no
        aparece, en la carpeta raiz del sujeto.
        """
        subject_dir  = self._subject_dir(num)
        empatica_dir = os.path.join(subject_dir, "Empatica_data")
        # Acepta también la variante con espacio ('Empatica data')
        if not os.path.isdir(empatica_dir):
            alt = os.path.join(subject_dir, "Empatica data")
            if os.path.isdir(alt):
                empatica_dir = alt

        session_csv = None

        # 1) Buscar session_times dentro de Empatica_data
        if os.path.isdir(empatica_dir):
            for entry in os.scandir(empatica_dir):
                if entry.is_file() and "session_times" in entry.name.lower():
                    session_csv = entry.path
                    break

        # 2) Si no está ahí, buscar en la carpeta raíz del sujeto
        if session_csv is None and os.path.isdir(subject_dir):
            for entry in os.scandir(subject_dir):
                if entry.is_file() and "session_times" in entry.name.lower():
                    session_csv = entry.path
                    break

        if session_csv is not None:
            self._fill_phase_entries_from_csv(session_csv)
            print(f"[OK] Session times (Empatica): {os.path.basename(session_csv)}")
            self._set_status(
                f"Subject {num} loaded\n{len(self._signals)} signal(s)\n"
                f"Session times from file."
            )
        else:
            print("[WARN] No session_times file found for this Empatica subject.")
            self._set_status(
                f"Subject {num} loaded\n{len(self._signals)} signal(s)\n"
                f"No session_times file — enter phases manually."
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
        subject_dir = self._subject_dir(num)
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
        """
        Deja la app como recien arrancada PERO conservando la sesion ya
        configurada (carpeta, dispositivo, senales y frecuencias originales).
        No guarda ningun dato del sujeto anterior. No hace falta volver a
        seleccionar la base de datos.
        """
        self._signals         = {}
        self._stress_map      = {}
        self._current_subject = None

        # Restaurar las frecuencias ORIGINALES del setup
        if self._fs_original:
            self._cfg["fs"] = dict(self._fs_original)

        # Limpiar el visor y dejar el mensaje inicial
        self._viewer._clear()
        ctk.CTkLabel(
            self._viewer._frame_plot,
            text="Load a subject to display signals.",
            text_color="gray", font=("Arial", 13),
        ).grid(row=0, column=0)

        # Limpiar la pestana de analisis
        self._analysis.reset_for_new_subject()

        # Vaciar los intervalos de fase
        for entry in (self._e_calm_s, self._e_calm_e,
                      self._e_stress_s, self._e_stress_e):
            entry.delete(0, "end")

        # Reconstruir los botones de sujeto
        if self._cfg:
            self._rebuild_subject_buttons()

        self._tabs.set("Signal Viewer")

        if self._cfg:
            self._set_status(
                f"App reset.\nSession: {self._cfg.get('device', '')}\n"
                f"Subjects: {self._cfg.get('n_subjects', 0)}\n"
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