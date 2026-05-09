"""
models.py
Feature extraction, stress classification (Random Forest), and ECG BPM.

Public API
----------
extract_features_windowed(signal, fs, window_sec, step_sec)
    → (features: np.ndarray, t_centers: np.ndarray)

build_labels(t_centers, intervals)
    → np.ndarray of 0/1

train_stress_model(X, y) → trained RandomForestClassifier
predict_stress(model, X) → np.ndarray of 0/1

compute_bpm(ecg_signal, fs) → (bpm_times, bpm_values)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ── Feature extraction ────────────────────────────────────────────────────────

def _window_features(segment: np.ndarray) -> np.ndarray:
    """Statistical + spectral features for one window of a 1-D signal."""
    seg = segment.astype(float)
    mean   = np.mean(seg)
    std    = np.std(seg)
    rms    = np.sqrt(np.mean(seg ** 2))
    ptp    = np.ptp(seg)
    skew   = float(np.mean(((seg - mean) / (std + 1e-8)) ** 3))
    kurt   = float(np.mean(((seg - mean) / (std + 1e-8)) ** 4))

    # Spectral power (fast via FFT)
    fft_mag = np.abs(np.fft.rfft(seg)) ** 2
    total_power = fft_mag.sum() + 1e-8
    low_power   = fft_mag[:len(fft_mag) // 4].sum() / total_power
    high_power  = fft_mag[len(fft_mag) // 4:].sum() / total_power

    return np.array([mean, std, rms, ptp, skew, kurt, low_power, high_power],
                    dtype=float)

FEATURE_NAMES = ["Mean", "Std", "RMS", "PtP", "Skewness",
                 "Kurtosis", "Low-freq power", "High-freq power"]


def extract_features_windowed(
    signal: np.ndarray,
    fs: float,
    window_sec: float = 60.0,
    step_sec: float   = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over *signal* and extract features per window.
    Returns (X [n_windows, n_features], t_centers [n_windows]) in seconds.
    """
    if signal.ndim == 2:
        signal = signal[:, 0]   # ECG: use first lead

    win_samples  = int(window_sec * fs)
    step_samples = int(step_sec   * fs)

    features, centers = [], []
    start = 0
    while start + win_samples <= len(signal):
        seg = signal[start : start + win_samples]
        features.append(_window_features(seg))
        centers.append((start + win_samples / 2) / fs)
        start += step_samples

    if not features:
        return np.empty((0, len(FEATURE_NAMES))), np.array([])

    return np.array(features), np.array(centers)


# ── Label builder ─────────────────────────────────────────────────────────────

def build_labels(
    t_centers: np.ndarray,
    stress_intervals: list[tuple[float, float]],
) -> np.ndarray:
    """
    Label each window center as stressed (1) or not (0).
    *stress_intervals*: list of (start_sec, end_sec) tuples.
    """
    labels = np.zeros(len(t_centers), dtype=int)
    for start, end in stress_intervals:
        labels[(t_centers >= start) & (t_centers <= end)] = 1
    return labels


# ── Model ─────────────────────────────────────────────────────────────────────

class StressModel:
    """Thin wrapper around RandomForestClassifier with per-signal scalers."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.clf    = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def feature_importances(self) -> np.ndarray:
        return self.clf.feature_importances_

    def save(self, path: str) -> None:
        joblib.dump({"clf": self.clf, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str) -> "StressModel":
        obj = cls()
        data = joblib.load(path)
        obj.clf    = data["clf"]
        obj.scaler = data["scaler"]
        obj.trained = True
        return obj


# ── BPM from ECG ─────────────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: float,
              low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_bpm(
    ecg_signal: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    step_sec: float   = 5.0,
    min_rr_sec: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks and compute sliding-window BPM.
    Returns (t_bpm [seconds], bpm_values).
    """
    if ecg_signal.ndim == 2:
        ecg_signal = ecg_signal[:, 1]   # lead II

    filtered   = _bandpass(ecg_signal, fs)
    min_dist   = int(min_rr_sec * fs)
    peaks, _   = find_peaks(filtered, distance=min_dist,
                             height=0.3 * np.std(filtered))

    if len(peaks) < 2:
        return np.array([]), np.array([])

    peak_times = peaks / fs
    win_samples  = int(window_sec * fs)
    step_samples = int(step_sec   * fs)

    bpm_times, bpm_values = [], []
    start = 0
    while start + win_samples <= len(ecg_signal):
        t_start = start / fs
        t_end   = (start + win_samples) / fs
        rr_peaks = peak_times[(peak_times >= t_start) & (peak_times < t_end)]
        if len(rr_peaks) >= 2:
            mean_rr = np.diff(rr_peaks).mean()
            bpm_values.append(60.0 / mean_rr)
            bpm_times.append((t_start + t_end) / 2)
        start += step_samples

    return np.array(bpm_times), np.array(bpm_values)
