"""
filters.py
Artifact removal and signal conditioning pipeline for BioVisor.

Pipeline per signal (applied in order):
  1. IQR outlier detection  → spike samples replaced by linear interpolation
  2. Bandpass filter         → remove physiologically impossible frequencies
  3. Savitzky-Golay smooth  → for slow signals (EDA, SKT, TEMP) only

All steps are configurable per signal type via config.SIGNAL_FILTER_PARAMS.
The raw signal is never modified in place — always returns a new array.

Public API
----------
clean_signal(signal, sig_type, fs) → (cleaned: np.ndarray, report: dict)
clean_all(signals, fs_map)         → (cleaned_signals, reports)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from app.core.config import SIGNAL_FILTER_PARAMS


# ── Step 1: IQR outlier removal ───────────────────────────────────────────────

def _remove_outliers_iqr(
    signal: np.ndarray,
    iqr_factor: float,
) -> tuple[np.ndarray, int]:
    """
    Detect samples outside  median ± iqr_factor × IQR.
    Detected outliers are replaced by linear interpolation from their neighbours.
    Returns (cleaned_signal, n_outliers_removed).
    """
    sig = signal.copy().astype(float)
    q1, q3 = np.percentile(sig, 25), np.percentile(sig, 75)
    iqr     = q3 - q1
    lo      = q1 - iqr_factor * iqr
    hi      = q3 + iqr_factor * iqr

    bad = np.where((sig < lo) | (sig > hi))[0]
    if len(bad) == 0:
        return sig, 0

    # Build interpolation from the good samples
    good = np.where((sig >= lo) & (sig <= hi))[0]
    if len(good) < 2:
        # Nothing good to interpolate from — just clip
        sig = np.clip(sig, lo, hi)
        return sig, len(bad)

    sig[bad] = np.interp(bad, good, sig[good])
    return sig, len(bad)


# ── Step 2: Bandpass filter ────────────────────────────────────────────────────

def _bandpass(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq    = fs / 2.0
    low_n  = max(low_hz  / nyq, 1e-4)
    high_n = min(high_hz / nyq, 0.9999)
    if low_n >= high_n:
        return signal.copy()
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


# ── Step 3: Savitzky-Golay smoothing ──────────────────────────────────────────

def _savgol(
    signal: np.ndarray,
    window_samples: int,
    poly_order: int,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing. Window is forced odd and > poly_order."""
    win = window_samples if window_samples % 2 == 1 else window_samples + 1
    win = max(win, poly_order + 2 if (poly_order + 2) % 2 == 1 else poly_order + 3)
    win = min(win, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if win < poly_order + 1 or win < 3:
        return signal.copy()
    return savgol_filter(signal, win, poly_order)


# ── Main public function ────────────────────────────────────────────────────────

def clean_signal(
    signal: np.ndarray,
    sig_type: str,
    fs: float,
) -> tuple[np.ndarray, dict]:
    """
    Apply the full artifact-removal pipeline for *sig_type*.
    Parameters are taken from config.SIGNAL_FILTER_PARAMS.

    Returns
    -------
    cleaned : np.ndarray
        Cleaned 1-D signal, same length as input.
    report : dict
        {
          "n_outliers"  : int,    samples replaced by IQR step
          "bandpass"    : bool,   whether bandpass was applied
          "smoothed"    : bool,   whether SavGol was applied
          "pct_removed" : float,  percentage of samples that were outliers
        }
    """
    sig = signal.ravel().astype(float)
    params = SIGNAL_FILTER_PARAMS.get(sig_type,
             SIGNAL_FILTER_PARAMS["_default"])

    report = {"n_outliers": 0, "bandpass": False,
              "smoothed": False, "pct_removed": 0.0}

    # ── Step 1: outlier removal ────────────────────────────────────────────
    iqr_factor = params.get("iqr_factor")
    if iqr_factor is not None:
        sig, n_out = _remove_outliers_iqr(sig, iqr_factor)
        report["n_outliers"]  = n_out
        report["pct_removed"] = 100.0 * n_out / max(len(sig), 1)

    # ── Step 2: bandpass ──────────────────────────────────────────────────
    bp = params.get("bandpass")
    if bp is not None and fs > 0:
        low_hz, high_hz = bp
        # Skip if Nyquist is too low for the requested band
        if high_hz < fs / 2:
            sig = _bandpass(sig, fs, low_hz, high_hz)
            report["bandpass"] = True

    # ── Step 3: Savitzky-Golay smooth ─────────────────────────────────────
    sg = params.get("savgol")
    if sg is not None:
        win, poly = sg
        sig = _savgol(sig, win, poly)
        report["smoothed"] = True

    return sig, report


def clean_all(
    signals: dict[str, np.ndarray],
    fs_map:  dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """
    Apply clean_signal() to every signal in the dict.
    Returns (cleaned_signals, reports) with the same keys.
    """
    cleaned: dict[str, np.ndarray] = {}
    reports: dict[str, dict]       = {}

    for sig_type, signal in signals.items():
        fs = fs_map.get(sig_type, 2000)
        cleaned[sig_type], reports[sig_type] = clean_signal(signal, sig_type, fs)
        r = reports[sig_type]
        print(
            f"[FILTER] {sig_type:6s}  outliers removed: {r['n_outliers']:5d} "
            f"({r['pct_removed']:.2f}%)  "
            f"bandpass: {'yes' if r['bandpass'] else 'no '}  "
            f"smooth: {'yes' if r['smoothed'] else 'no'}"
        )

    return cleaned, reports