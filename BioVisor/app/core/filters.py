"""
filters.py
Artifact removal and signal conditioning pipeline for BioVisor.

Pipeline per signal (applied in order):
  1. Sample-jump detection  → consecutive samples that change impossibly fast → NaN
  2. NaN interpolation      → all NaN gaps filled by linear interpolation

Artifact detection uses a RELATIVE threshold: a sample-to-sample change
larger than N × median(|diff|) is marked as artifact and interpolated.
This approach is unit-independent, so it works regardless of whether
Biopac exports in mV, µV, V, or any other unit.

Public API
----------
clean_signal(signal, sig_type, fs) → (cleaned: np.ndarray, report: dict)
clean_all(signals, fs_map)         → (cleaned_signals, reports)
"""

from __future__ import annotations
import numpy as np


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

ABS_MIN: dict[str, float | None] = {
    "EDA":  0.0,
    "GSR":  0.0,
    "SKT":  15.0,   # skin temp can't be below 15°C
    "TEMP": 15.0,
    "HR":   0.0,
}


def _mark_artifacts(signal: np.ndarray, sig_type: str) -> tuple[np.ndarray, int]:
    """Mark artifact samples as NaN using relative jump detection."""
    sig  = signal.copy().astype(float)
    mask = np.zeros(len(sig), dtype=bool)

    # Flag values below physical minimum (e.g. EDA can't be negative)
    floor = ABS_MIN.get(sig_type)
    if floor is not None:
        mask |= sig < floor

    # Flag samples where the change vs previous sample is abnormally large
    if len(sig) > 1:
        diffs     = np.abs(np.diff(sig))
        median_d  = np.median(diffs[diffs > 0]) if diffs.any() else 1.0
        threshold = JUMP_FACTOR.get(sig_type, JUMP_FACTOR["_default"]) * median_d
        jump      = diffs > threshold
        mask[:-1] |= jump
        mask[1:]  |= jump

    sig[mask] = np.nan
    return sig, int(mask.sum())


def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Fill NaN gaps with linear interpolation."""
    sig  = signal.copy()
    nans = np.isnan(sig)
    if not nans.any():
        return sig
    good = ~nans
    if good.sum() < 2:
        sig[:] = 0.0
        return sig
    idx = np.arange(len(sig))
    sig[nans] = np.interp(idx[nans], idx[good], sig[good])
    return sig


def clean_signal(signal: np.ndarray, sig_type: str, fs: float) -> tuple[np.ndarray, dict]:
    """Apply artifact removal to a signal. Returns (cleaned, report)."""
    sig, n_art = _mark_artifacts(signal.ravel().astype(float), sig_type)
    sig = _interpolate_nans(sig)
    return sig, {"n_artifacts": n_art, "pct_artifacts": 100.0 * n_art / max(len(sig), 1)}


def clean_all(signals: dict[str, np.ndarray], fs_map: dict[str, float]) -> tuple[dict, dict]:
    """Apply clean_signal() to every signal. Returns (cleaned, reports)."""
    cleaned, reports = {}, {}
    for sig, arr in signals.items():
        cleaned[sig], reports[sig] = clean_signal(arr, sig, fs_map.get(sig, 2000))
        r = reports[sig]
        print(f"[FILTER] {sig:6s}  artifacts: {r['n_artifacts']:6d} ({r['pct_artifacts']:.3f}%)")
    return cleaned, reports