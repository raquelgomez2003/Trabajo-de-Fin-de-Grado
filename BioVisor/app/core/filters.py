"""
filters.py
Artifact removal and signal conditioning pipeline for BioVisor.

Pipeline per signal (applied in order):
  1. Sample-jump detection  → consecutive samples that change impossibly fast → NaN
  2. NaN interpolation      → all NaN gaps filled by linear interpolation
  3. Bandpass filter        → remove physiologically impossible frequencies
  4. Savitzky-Golay smooth  → only for very slow signals (EDA, SKT, TEMP)

NOTE on unit-agnostic design:
  Biopac can export in mV, µV, V, or arbitrary units depending on the amplifier
  and export settings. Fixed physiological ranges (e.g. ECG must be ±5 mV) would
  silently zero out entire signals if the unit is different.

  Instead, jump detection uses a RELATIVE threshold: a sample-to-sample change
  larger than N × median(|diff|) is an artifact. This works regardless of units.
  Only EDA uses an absolute floor (can't be negative in any unit system).

Public API
----------
clean_signal(signal, sig_type, fs) → (cleaned: np.ndarray, report: dict)
clean_all(signals, fs_map)         → (cleaned_signals, reports)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from app.core.config import SIGNAL_FILTER_PARAMS


# ── Jump thresholds (× median absolute diff) ─────────────────────────────────
# A sample-to-sample change larger than JUMP_FACTOR × median|diff| = artifact.
# These are unit-independent.
# Higher = less aggressive. EMG/ECG need high values (real peaks look like jumps).
# EDA/SKT/RESP change slowly so a moderate factor catches glitches.
JUMP_FACTOR: dict[str, float] = {
    "ECG":  500.0,  # R-peaks cause huge jumps — only catch true glitches
    "BVP":  500.0,
    "PPG":  500.0,
    "EMG":  200.0,
    "EDA":  50.0,   # slow signal
    "GSR":  50.0,
    "RESP": 200.0,
    "SKT":  20.0,   # temperature never changes fast
    "TEMP": 20.0,
    "ACC":  200.0,
    "HR":   30.0,
    "_default": 200.0,
}

# Absolute minimum value — only for signals that physically can't go below 0
ABS_MIN: dict[str, float | None] = {
    "EDA":  0.0,
    "GSR":  0.0,
    "SKT":  15.0,   # skin temp can't be below 15°C
    "TEMP": 15.0,
    "HR":   0.0,
}


# ── Step 1: sample-jump artifact detection ────────────────────────────────────

def _mark_artifacts(
    signal: np.ndarray,
    sig_type: str,
) -> tuple[np.ndarray, int]:
    """
    Mark artifact samples as NaN using relative jump detection.
    Returns (signal_with_nans, n_artifacts).
    """
    sig  = signal.copy().astype(float)
    mask = np.zeros(len(sig), dtype=bool)

    # Absolute floor (only where physically meaningful)
    floor = ABS_MIN.get(sig_type)
    if floor is not None:
        mask |= sig < floor

    # Relative jump detection
    if len(sig) > 1:
        diffs      = np.abs(np.diff(sig))
        median_d   = np.median(diffs[diffs > 0]) if (diffs > 0).any() else 1.0
        factor     = JUMP_FACTOR.get(sig_type, JUMP_FACTOR["_default"])
        threshold  = factor * median_d

        jump       = diffs > threshold
        # Flag the sample after the jump and the one before
        jump_mask           = np.concatenate([[False], jump])
        jump_mask[:-1]     |= jump
        mask               |= jump_mask

    n_art     = int(mask.sum())
    sig[mask] = np.nan
    return sig, n_art


# ── Step 2: interpolate NaN gaps ─────────────────────────────────────────────

def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    sig  = signal.copy()
    nans = np.isnan(sig)
    if not nans.any():
        return sig
    idx  = np.arange(len(sig))
    good = ~nans
    if good.sum() < 2:
        sig[:] = 0.0
        return sig
    sig[nans] = np.interp(idx[nans], idx[good], sig[good])
    return sig


# ── Step 3: bandpass ──────────────────────────────────────────────────────────

def _bandpass(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyq    = fs / 2.0
    low_n  = max(low_hz  / nyq, 1e-4)
    high_n = min(high_hz / nyq, 0.9999)
    if low_n >= high_n:
        return signal.copy()
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


# ── Step 4: Savitzky-Golay ────────────────────────────────────────────────────

def _savgol(signal: np.ndarray, window_samples: int, poly_order: int) -> np.ndarray:
    win = window_samples if window_samples % 2 == 1 else window_samples + 1
    win = max(win, poly_order + 2 if (poly_order + 2) % 2 == 1 else poly_order + 3)
    win = min(win, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if win < poly_order + 1 or win < 3:
        return signal.copy()
    return savgol_filter(signal, win, poly_order)


# ── Main public functions ──────────────────────────────────────────────────────

def clean_signal(
    signal: np.ndarray,
    sig_type: str,
    fs: float,
) -> tuple[np.ndarray, dict]:
    """
    Apply the artifact-removal pipeline for *sig_type*.
    Returns (cleaned, report).
    """
    sig    = signal.ravel().astype(float)
    params = SIGNAL_FILTER_PARAMS.get(sig_type, SIGNAL_FILTER_PARAMS["_default"])
    report = {"n_artifacts": 0, "pct_artifacts": 0.0,
              "bandpass": False, "smoothed": False}

    # 1. Mark artifacts
    sig, n_art = _mark_artifacts(sig, sig_type)
    report["n_artifacts"]   = n_art
    report["pct_artifacts"] = 100.0 * n_art / max(len(sig), 1)

    # 2. Interpolate NaNs
    sig = _interpolate_nans(sig)

    # 3. Bandpass
    bp = params.get("bandpass")
    if bp is not None and fs > 0:
        low_hz, high_hz = bp
        if high_hz < fs / 2:
            sig = _bandpass(sig, fs, low_hz, high_hz)
            report["bandpass"] = True

    # 4. Savitzky-Golay
    sg = params.get("savgol")
    if sg is not None:
        sig = _savgol(sig, *sg)
        report["smoothed"] = True

    return sig, report


def clean_all(
    signals: dict[str, np.ndarray],
    fs_map:  dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Apply clean_signal() to every signal. Returns (cleaned, reports)."""
    cleaned: dict[str, np.ndarray] = {}
    reports: dict[str, dict]       = {}

    for sig_type, signal in signals.items():
        fs = fs_map.get(sig_type, 2000)
        cleaned[sig_type], reports[sig_type] = clean_signal(signal, sig_type, fs)
        r = reports[sig_type]
        print(
            f"[FILTER] {sig_type:6s}  artifacts: {r['n_artifacts']:6d} "
            f"({r['pct_artifacts']:.3f}%)  "
            f"bandpass: {'yes' if r['bandpass'] else 'no '}  "
            f"smooth: {'yes' if r['smoothed'] else 'no'}"
        )

    return cleaned, reports