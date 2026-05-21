"""
setup_window.py
Initial setup wizard — folder picker, device type, subject count,
signal selection, and sampling frequencies.
"""

from __future__ import annotations
import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

from app.core.config import DEVICE_SIGNALS, DEVICE_FS


class SetupWindow(ctk.CTkToplevel):

    def __init__(self, parent, on_confirm):
        super().__init__(parent)
        self.title("BioVisor — Setup")
        self.geometry("560x680")
        self.resizable(False, False)
        self.grab_set()

        self._on_confirm = on_confirm
        self._folder_var = ctk.StringVar(value="")
        self._device_var = ctk.StringVar(value="Biopac")
        self._n_subj_var = ctk.StringVar(value="5")
        self._sig_vars: dict[str, ctk.BooleanVar] = {}
        self._fs_vars:  dict[str, ctk.StringVar]  = {}
        self._build()

    def _build(self):
        pad = {"padx": 20, "pady": 6}

        ctk.CTkLabel(self, text="BioVisor — Session Setup",
                     font=("Arial", 17, "bold")).pack(pady=(18, 4))
        ctk.CTkLabel(self, text="Configure your dataset before loading signals.",
                     text_color="gray").pack(pady=(0, 12))

        # Folder picker
        frame_folder = ctk.CTkFrame(self)
        frame_folder.pack(fill="x", **pad)
        ctk.CTkLabel(frame_folder, text="Dataset folder:").pack(anchor="w", padx=10, pady=(8, 2))
        row = ctk.CTkFrame(frame_folder, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkEntry(row, textvariable=self._folder_var, width=350).pack(side="left")
        ctk.CTkButton(row, text="Browse…", width=90,
                      command=self._browse).pack(side="left", padx=6)

        # Device type
        frame_device = ctk.CTkFrame(self)
        frame_device.pack(fill="x", **pad)
        ctk.CTkLabel(frame_device, text="Device type:").pack(anchor="w", padx=10, pady=(8, 2))
        row_d = ctk.CTkFrame(frame_device, fg_color="transparent")
        row_d.pack(fill="x", padx=10, pady=(0, 8))
        for dev in ("Biopac", "Empatica"):
            ctk.CTkRadioButton(row_d, text=dev, variable=self._device_var,
                               value=dev, command=self._refresh_signals).pack(side="left", padx=10)

        # Number of subjects
        frame_subj = ctk.CTkFrame(self)
        frame_subj.pack(fill="x", **pad)
        row_s = ctk.CTkFrame(frame_subj, fg_color="transparent")
        row_s.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(row_s, text="Number of subjects:").pack(side="left")
        ctk.CTkEntry(row_s, textvariable=self._n_subj_var, width=60).pack(side="left", padx=10)

        # Signals + Fs
        ctk.CTkLabel(self, text="Signals to load  ·  Sampling frequency (Hz)",
                     font=("Arial", 12, "bold")).pack(anchor="w", padx=22, pady=(8, 0))
        self._frame_signals = ctk.CTkScrollableFrame(self, height=240)
        self._frame_signals.pack(fill="x", padx=20, pady=6)
        self._refresh_signals()

        ctk.CTkButton(self, text="Confirm & Open Viewer",
                      height=40, command=self._confirm).pack(pady=14)

    def _browse(self):
        folder = filedialog.askdirectory(title="Select dataset root folder")
        if folder:
            self._folder_var.set(folder)

    def _refresh_signals(self):
        for w in self._frame_signals.winfo_children():
            w.destroy()
        self._sig_vars.clear()
        self._fs_vars.clear()

        device     = self._device_var.get()
        default_fs = DEVICE_FS.get(device, {})

        # Header
        hdr = ctk.CTkFrame(self._frame_signals, fg_color="transparent")
        hdr.pack(fill="x", padx=4, pady=(0, 4))
        for text, width in (("Signal", 120), ("Include", 70), ("Fs (Hz)", 100)):
            ctk.CTkLabel(hdr, text=text, width=width,
                         font=("Arial", 11, "bold")).pack(side="left")

        for sig in DEVICE_SIGNALS.get(device, []):
            row = ctk.CTkFrame(self._frame_signals, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=2)
            self._sig_vars[sig] = ctk.BooleanVar(value=True)
            self._fs_vars[sig]  = ctk.StringVar(value=str(default_fs.get(sig, 2000)))
            ctk.CTkLabel(row, text=sig, width=120).pack(side="left")
            ctk.CTkCheckBox(row, variable=self._sig_vars[sig], text="", width=70).pack(side="left")
            ctk.CTkEntry(row, textvariable=self._fs_vars[sig], width=90).pack(side="left", padx=6)

    def _confirm(self):
        # Validate folder
        folder = self._folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Invalid folder", "Please select a valid dataset folder.")
            return

        # Validate subject count
        try:
            n_subjects = int(self._n_subj_var.get())
            assert n_subjects >= 1
        except Exception:
            messagebox.showerror("Invalid input", "Number of subjects must be a positive integer.")
            return

        # Validate signal selection
        selected = [sig for sig, var in self._sig_vars.items() if var.get()]
        if not selected:
            messagebox.showerror("No signals", "Select at least one signal to load.")
            return

        # Validate sampling frequencies
        fs_map = {}
        for sig in selected:
            try:
                fs_map[sig] = float(self._fs_vars[sig].get())
            except Exception:
                messagebox.showerror("Invalid Fs", f"Sampling frequency for {sig} is not valid.")
                return

        self.destroy()
        self._on_confirm({
            "folder":     folder,
            "device":     self._device_var.get(),
            "n_subjects": n_subjects,
            "signals":    selected,
            "fs":         fs_map,
        })