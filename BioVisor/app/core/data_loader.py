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

from app.core.config import DEVICE_SIGNALS, ECG_COLUMNS


# ── Type detection ────────────────────────────────────────────────────────────

def detect_signal_type(filename: str, device: str) -> str | None:
    """Return signal type if filename matches a known signal, else None."""
    name = os.path.splitext(filename)[0].upper()
    return next((sig for sig in DEVICE_SIGNALS.get(device, []) if sig in name), None)


# ── Per-device loaders ────────────────────────────────────────────────────────

def _load_biopac_csv(path: str) -> np.ndarray | None:
    """Load a Biopac CSV. Tries common separators and picks the most variable column."""
    for sep in (",", ";", "\t", " "):
        try:
            df = pd.read_csv(path, sep=sep, header=None, engine="python")
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.empty:
                continue
            col = int(df.var().idxmax()) if df.shape[1] > 1 else 0
            arr = df.iloc[:, col].values.astype(float)
            if len(arr) > 10:
                return arr
        except Exception:
            continue
    print(f"[WARN] Could not parse {path}")
    return None


def _load_empatica_csv(path: str) -> np.ndarray | None:
    """Load an Empatica E4 CSV. Skips the first two rows (timestamp + sample rate)."""
    try:
        raw = np.loadtxt(path, delimiter=",")
        return raw[2:] if raw.ndim == 1 else raw[2:, 0]
    except Exception as e:
        print(f"[WARN] Could not load Empatica {path}: {e}")
        return None


# ── Main loader ───────────────────────────────────────────────────────────────

def load_subject_folder(
    folder: str,
    device: str,
    selected_signals: list[str],
    fs_map: dict[str, float] | None = None,
    apply_filters: bool = True,
) -> dict[str, np.ndarray]:
    """
    Scan folder (and one level of subfolders) for CSV signal files.
    Returns {signal_type: 1D ndarray}.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Collect CSVs from root + immediate subfolders
    csv_files = [
        entry.path
        for entry in os.scandir(folder)
        if entry.is_file() and entry.name.lower().endswith(".csv")
    ] + [
        sub.path
        for entry in os.scandir(folder) if entry.is_dir()
        for sub in os.scandir(entry.path)
        if sub.is_file() and sub.name.lower().endswith(".csv")
    ]

    signals: dict[str, np.ndarray] = {}
    loader = _load_biopac_csv if device == "Biopac" else _load_empatica_csv

    for full_path in csv_files:
        filename = os.path.basename(full_path)
        sig_type = detect_signal_type(filename, device)
        if sig_type is None or sig_type not in selected_signals or sig_type in signals:
            continue

        data = loader(full_path)
        if data is not None and len(data) > 10:
            signals[sig_type] = data
            print(f"[OK] {sig_type}: {len(data)} samples  ← {filename}")

    if apply_filters and signals:
        from app.core.filters import clean_all
        signals, _ = clean_all(signals, fs_map or {})

    return signals


# ── Utility ───────────────────────────────────────────────────────────────────

def time_axis(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return time axis in seconds for a signal array."""
    return np.arange(signal.shape[0]) / fs