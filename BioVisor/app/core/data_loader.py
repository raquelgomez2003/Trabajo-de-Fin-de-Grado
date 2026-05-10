"""
data_loader.py
Signal loading for Biopac and Empatica datasets.

Public API
----------
detect_signal_type(filename, device)  → str | None
load_subject_folder(folder, device, selected_signals, fs_map, apply_filters)
    → dict[str, np.ndarray]
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional

from app.core.config import DEVICE_SIGNALS, ECG_COLUMNS


# ── Type detection ─────────────────────────────────────────────────────────────

def detect_signal_type(filename: str, device: str) -> Optional[str]:
    """Return signal type string if filename matches a known signal, else None."""
    name_upper = os.path.splitext(filename)[0].upper()
    for sig in DEVICE_SIGNALS.get(device, []):
        if sig in name_upper:
            return sig
    return None


# ── Per-device loaders ─────────────────────────────────────────────────────────

def _load_biopac_csv(path: str, sig_type: str) -> Optional[np.ndarray]:
    """
    Robustly load a Biopac CSV.
    Handles:
      - Single numeric column (all signals including ECG)
      - Optional text header row (skipped automatically)
      - Comma or semicolon separators
    Returns a 1-D numpy array of float values.
    """
    try:
        # Try reading with pandas — it handles headers and separators well
        for sep in (",", ";", "\t", " "):
            try:
                df = pd.read_csv(path, sep=sep, header=None, engine="python")
                # Drop any rows that are not fully numeric
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                if df.empty:
                    continue
                # If multiple columns, pick the one with highest variance (most likely ECG lead)
                if df.shape[1] > 1:
                    col = int(df.var().idxmax())
                    arr = df.iloc[:, col].values.astype(float)
                else:
                    arr = df.iloc[:, 0].values.astype(float)
                if len(arr) > 10:
                    return arr
            except Exception:
                continue
        print(f"[WARN] Could not parse {path}")
        return None
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def _load_empatica_csv(path: str, sig_type: str) -> Optional[np.ndarray]:
    """
    Empatica E4 CSVs: first row = Unix timestamp, second row = sample rate,
    remaining rows = samples (single column).
    """
    try:
        raw = np.loadtxt(path, delimiter=",")
        # Skip header rows if present
        if raw.ndim == 1:
            return raw[2:]   # skip timestamp + fs rows
        return raw[2:, 0]
    except Exception as e:
        print(f"[WARN] Could not load Empatica {path}: {e}")
        return None


# ── Main loader ────────────────────────────────────────────────────────────────

def load_subject_folder(
    folder: str,
    device: str,
    selected_signals: list[str],
    fs_map: dict[str, float] | None = None,
    apply_filters: bool = True,
) -> dict[str, np.ndarray]:
    """
    Scan *folder* (and one level of subfolders) for CSV signal files.
    Handles the structure: Base1_SujetoN/Biopac data/SubjectXX_ECG.csv

    Parameters
    ----------
    apply_filters : if True, runs the artifact-removal pipeline after loading.
    fs_map        : sampling frequencies per signal (needed for bandpass filter).

    Returns {signal_type: 1D ndarray}.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Collect all CSV files: root folder + immediate subfolders
    csv_files: list[str] = []
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.lower().endswith(".csv"):
            csv_files.append(entry.path)
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and sub.name.lower().endswith(".csv"):
                    csv_files.append(sub.path)

    signals: dict[str, np.ndarray] = {}

    for full_path in csv_files:
        filename = os.path.basename(full_path)
        sig_type = detect_signal_type(filename, device)
        if sig_type is None or sig_type not in selected_signals:
            continue
        if sig_type in signals:
            continue

        if device == "Biopac":
            data = _load_biopac_csv(full_path, sig_type)
        else:
            data = _load_empatica_csv(full_path, sig_type)

        if data is not None and len(data) > 10:
            signals[sig_type] = data
            print(f"[OK] {sig_type}: {len(data)} samples  ← {filename}")

    # ── Artifact removal ───────────────────────────────────────────────────
    if apply_filters and signals:
        from app.core.filters import clean_all
        signals, _ = clean_all(signals, fs_map or {})

    return signals


# ── Utility ────────────────────────────────────────────────────────────────────

def time_axis(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return time axis in seconds for a 1-D or 2-D signal array."""
    n = signal.shape[0]
    return np.arange(n) / fs