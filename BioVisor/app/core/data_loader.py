"""
data_loader.py
Signal loading for Biopac and Empatica datasets.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from app.core.config import DEVICE_SIGNALS, ECG_COLUMNS


def detect_signal_type(filename: str, device: str) -> str | None:
    name = os.path.splitext(filename)[0].upper()
    return next((sig for sig in DEVICE_SIGNALS.get(device, []) if sig in name), None)


def _load_biopac_csv(path: str) -> np.ndarray | None:
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


def _load_empatica_csv(
    path: str,
    sig_type: str,
    fs_map: dict[str, float],
) -> np.ndarray | None:
    """
    Empatica E4 format:
      Row 0: Unix timestamp(s)
      Row 1: sample rate(s) in Hz
      Row 2+: data
    ACC has 3 columns → returns magnitude.
    """
    filename = os.path.basename(path)
    try:
        with open(path, "r") as f:
            lines = f.readlines()

        if len(lines) < 3:
            return None

        # Row 1: sample rate (first value)
        row1 = lines[1].strip().split(",")
        try:
            fs = float(row1[0])
        except ValueError:
            fs = 1.0

        # Data rows from row 2 onwards
        rows = []
        for line in lines[2:]:
            vals = line.strip().split(",")
            try:
                rows.append([float(v) for v in vals if v.strip()])
            except ValueError:
                continue

        if not rows:
            return None

        arr = np.array(rows)

        # ACC: 3 columns → magnitude √(x²+y²+z²)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            signal = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
        elif arr.ndim == 2:
            signal = arr[:, 0]
        else:
            signal = arr

        fs_map[sig_type] = fs
        return signal.astype(float)

    except Exception as e:
        print(f"[WARN] Empatica {filename}: {e}")
        return None


def load_subject_folder(
    folder: str,
    device: str,
    selected_signals: list[str],
    fs_map: dict[str, float] | None = None,
    apply_filters: bool = True,
) -> dict[str, np.ndarray]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    if fs_map is None:
        fs_map = {}

    # Collect all CSV files in this folder and one level of subfolders
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

    print(f"[INFO] Device={device}, folder={folder}")
    print(f"[INFO] Found {len(csv_files)} CSV files")

    signals: dict[str, np.ndarray] = {}

    for full_path in csv_files:
        filename  = os.path.basename(full_path)
        sig_type  = detect_signal_type(filename, device)

        if sig_type is None or sig_type not in selected_signals or sig_type in signals:
            continue

        print(f"[LOAD] {filename} → {sig_type}")

        if device == "Biopac":
            data = _load_biopac_csv(full_path)
        else:
            data = _load_empatica_csv(full_path, sig_type, fs_map)

        if data is not None and len(data) > 10:
            signals[sig_type] = data
            print(f"[OK]   {sig_type}: {len(data)} samples @ {fs_map.get(sig_type, '?')} Hz")
        else:
            print(f"[FAIL] {sig_type} ← {filename}")

    if apply_filters and signals:
        from app.core.filters import clean_all
        signals, _ = clean_all(signals, fs_map)

    return signals


def time_axis(signal: np.ndarray, fs: float) -> np.ndarray:
    return np.arange(signal.shape[0]) / fs