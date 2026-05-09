"""
data_loader.py
Signal loading for Biopac and Empatica datasets.

Public API
----------
detect_signal_type(filename, device)  → str | None
load_subject_folder(folder, device, selected_signals) → dict[str, np.ndarray]
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
    try:
        if sig_type == "ECG":
            df = pd.read_csv(path, header=0, names=ECG_COLUMNS)
            return df.values.astype(float)
        else:
            return np.loadtxt(path, delimiter=",", skiprows=1)
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
) -> dict[str, np.ndarray]:
    """
    Scan *folder* for CSV files belonging to *device*.
    Only loads signals that are in *selected_signals*.
    Returns {signal_type: ndarray}.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    signals: dict[str, np.ndarray] = {}

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".csv"):
            continue
        sig_type = detect_signal_type(filename, device)
        if sig_type is None or sig_type not in selected_signals:
            continue

        full_path = os.path.join(folder, filename)
        if device == "Biopac":
            data = _load_biopac_csv(full_path, sig_type)
        else:
            data = _load_empatica_csv(full_path, sig_type)

        if data is not None and len(data) > 0:
            signals[sig_type] = data

    return signals


# ── Utility ────────────────────────────────────────────────────────────────────

def time_axis(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return time axis in seconds for a 1-D or 2-D signal array."""
    n = signal.shape[0]
    return np.arange(n) / fs
