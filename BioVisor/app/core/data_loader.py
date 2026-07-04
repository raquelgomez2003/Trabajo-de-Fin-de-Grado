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


def _resample_to_target(signal: np.ndarray, fs_orig: float,
                         fs_target: float = 2000.0) -> np.ndarray:
    """Resample signal from fs_orig to fs_target using linear interpolation.

    La duracion real (len / fs_orig) se conserva: el numero de muestras
    de salida es round(duracion * fs_target), asi que el eje de tiempo
    posterior (len / fs_target) coincide con la duracion original.
    """
    if abs(fs_orig - fs_target) < 0.01:
        return signal.astype(np.float32)
    duration   = len(signal) / fs_orig
    n_orig     = len(signal)
    n_target   = int(round(duration * fs_target))
    if n_target < 2 or n_orig < 2:
        return signal.astype(np.float32)
    t_orig     = np.linspace(0, duration, n_orig,   endpoint=False)
    t_target   = np.linspace(0, duration, n_target, endpoint=False)
    return np.interp(t_target, t_orig, signal).astype(np.float32)


def _load_empatica_csv(
    path: str,
    sig_type: str,
    fs_map: dict[str, float],
) -> np.ndarray | None:
    """
    Empatica combinado (formato SujetoN_*.csv):
      - BVP / EDA / HR / TEMP: un valor por linea, SIN cabecera ni metadatos.
      - ACC: cabecera 'ACC_X,ACC_Y,ACC_Z,start_time_unix,sampling_rate' +
        filas de datos -> se devuelve la magnitud sqrt(x^2 + y^2 + z^2).

    La fs se toma del popup/config (fs_map[sig_type]) y la senal se MANTIENE a
    su frecuencia nativa (BVP 64, EDA 4, HR 1, TEMP 4, ACC 32 Hz). NO se
    remuestrea a 2000 Hz: para Empatica eso solo multiplica el tamano y destroza
    las features de las senales lentas.
    """
    filename = os.path.basename(path)
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if not lines:
            return None

        fs_orig = fs_map.get(sig_type)
        if fs_orig is None or fs_orig <= 0:
            print(f"[WARN] Empatica {filename}: no hay fs para '{sig_type}' "
                  f"en el popup de carga; se omite esta senal.")
            return None

        # Saltar una cabecera de texto si la hay (p. ej. ACC_X,ACC_Y,...)
        start = 0
        first = lines[0].strip()
        if first and any(c.isalpha() for c in first):
            start = 1

        rows = []
        for line in lines[start:]:
            vals = line.strip().split(",")
            try:
                rows.append([float(v) for v in vals if v.strip()])
            except ValueError:
                continue
        if not rows:
            return None

        arr = np.array(rows)

        # ACC: >=3 columnas -> magnitud con las 3 primeras (x, y, z)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            signal = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        elif arr.ndim == 2:
            signal = arr[:, 0]
        else:
            signal = arr

        signal = np.asarray(signal, dtype=float).ravel()
        if signal.size < 2:
            return None

        dur = signal.size / fs_orig
        print(f"[EMPATICA] {sig_type:5s}: {signal.size} muestras @ {fs_orig} Hz "
              f"(nativa) -> duracion {dur:.1f}s (sin remuestreo)")

        fs_map[sig_type] = float(fs_orig)   # se mantiene la fs nativa
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

        print(f"[LOAD] {filename} -> {sig_type}")

        if device == "Biopac":
            data = _load_biopac_csv(full_path)
        else:
            data = _load_empatica_csv(full_path, sig_type, fs_map)

        if data is not None and len(data) > 10:
            signals[sig_type] = data
            print(f"[OK]   {sig_type}: {len(data)} samples @ {fs_map.get(sig_type, '?')} Hz")
        else:
            print(f"[FAIL] {sig_type} <- {filename}")

    if apply_filters and signals:
        from app.core.filters import clean_all
        signals, _ = clean_all(signals, fs_map)

    return signals


def time_axis(signal: np.ndarray, fs: float) -> np.ndarray:
    return np.arange(signal.shape[0]) / fs