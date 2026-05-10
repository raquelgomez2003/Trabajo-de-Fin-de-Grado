"""
models.py
Feature extraction, stress classification, ECG BPM and RR interval analysis.

Key improvements:
  - Short windows (10 s / 5 s step) for finer temporal resolution
  - RR-interval series computed beat-by-beat (not windowed average)
  - Full HRV features: RMSSD, SDNN, pNN50, LF/HF ratio
  - Gradient Boosting classifier (better than plain RF on small datasets)
  - Multi-signal physiological labelling with weighted majority vote
  - Two stress estimations:
      * StressModel  → ML-based prediction per signal
      * compute_rr   → direct RR-based detection (no ML needed)

Public API
----------
compute_rr(ecg, fs)                     → (rr_times, rr_ms)
compute_bpm(ecg, fs, ...)               → (t_bpm, bpm_values)
extract_features_windowed(...)          → (X, t_centers)
build_physiological_labels(...)         → (t_centers, y)
build_labels_from_intervals(...)        → y
class StressModel
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: float,
              low: float = 0.5, high: float = 40.0) -> np.ndarray:
    nyq    = fs / 2.0
    low_n  = max(low  / nyq, 1e-4)
    high_n = min(high / nyq, 0.9999)
    if low_n >= high_n:
        return signal.copy()
    b, a = butter(4, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def _detect_peaks(ecg_1d: np.ndarray, fs: float,
                  min_rr_sec: float = 0.3) -> np.ndarray:
    filtered = _bandpass(ecg_1d, fs, 0.5, 40.0)
    peaks, _ = find_peaks(filtered,
                          distance=int(min_rr_sec * fs),
                          height=0.3 * np.std(filtered))
    return peaks


# ── RR interval series ────────────────────────────────────────────────────────

def compute_rr(
    ecg_signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute beat-by-beat RR intervals from an ECG signal.

    Returns
    -------
    rr_times : time of each interval midpoint (seconds)
    rr_ms    : RR interval duration (milliseconds)
    """
    ecg_1d     = ecg_signal.ravel()
    peaks      = _detect_peaks(ecg_1d, fs)
    if len(peaks) < 2:
        return np.array([]), np.array([])
    peak_times = peaks / fs
    rr_ms      = np.diff(peak_times) * 1000.0
    rr_times   = (peak_times[:-1] + peak_times[1:]) / 2.0
    return rr_times, rr_ms


# ── Windowed BPM ──────────────────────────────────────────────────────────────

def compute_bpm(
    ecg_signal: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    step_sec:   float = 5.0,
    min_rr_sec: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window BPM. Returns (t_bpm, bpm_values)."""
    ecg_1d = ecg_signal.ravel()
    peaks  = _detect_peaks(ecg_1d, fs, min_rr_sec)
    if len(peaks) < 2:
        return np.array([]), np.array([])

    peak_times   = peaks / fs
    win_samples  = int(window_sec * fs)
    step_samples = max(1, int(step_sec * fs))

    t_out, bpm_out = [], []
    start = 0
    while start + win_samples <= len(ecg_1d):
        t0 = start / fs
        t1 = (start + win_samples) / fs
        pp = peak_times[(peak_times >= t0) & (peak_times < t1)]
        if len(pp) >= 2:
            bpm_out.append(60.0 / np.diff(pp).mean())
            t_out.append((t0 + t1) / 2)
        start += step_samples

    return np.array(t_out), np.array(bpm_out)


# ── HRV features for one ECG window ──────────────────────────────────────────

def _hrv_features(segment: np.ndarray, fs: float) -> np.ndarray:
    """8 clinically validated HRV features. Returns zeros if not enough peaks."""
    feats = np.zeros(8, dtype=float)
    try:
        peaks = _detect_peaks(segment, fs)
        if len(peaks) < 3:
            return feats
        rr      = np.diff(peaks) / fs * 1000.0   # ms
        diff_rr = np.diff(rr)

        feats[0] = np.mean(rr)                                     # mean RR
        feats[1] = np.std(rr)                                      # SDNN
        feats[2] = np.sqrt(np.mean(diff_rr ** 2))                  # RMSSD
        feats[3] = 60000.0 / (np.mean(rr) + 1e-10)                 # mean BPM
        feats[4] = np.sum(np.abs(diff_rr) > 50) / max(len(diff_rr), 1)  # pNN50

        if len(rr) >= 8:
            freqs   = np.fft.rfftfreq(len(rr), d=np.mean(rr) / 1000.0)
            power   = np.abs(np.fft.rfft(rr - rr.mean())) ** 2
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.40)
            lf_p    = power[lf_mask].sum() if lf_mask.any() else 0.0
            hf_p    = power[hf_mask].sum() if hf_mask.any() else 1e-10
            feats[5] = lf_p
            feats[6] = hf_p
            feats[7] = lf_p / (hf_p + 1e-10)   # LF/HF — key stress marker
    except Exception:
        pass
    return feats


HRV_FEATURE_NAMES = [
    "Mean RR (ms)", "SDNN (ms)", "RMSSD (ms)", "Mean BPM",
    "pNN50", "LF power", "HF power", "LF/HF ratio",
]


# ── Per-window features per signal type ──────────────────────────────────────

def _window_features(segment: np.ndarray, fs: float, sig_type: str) -> np.ndarray:
    seg  = segment.astype(float)
    mean = np.mean(seg)
    std  = np.std(seg) + 1e-10
    rms  = np.sqrt(np.mean(seg ** 2))
    ptp  = np.ptp(seg)
    skew = float(np.mean(((seg - mean) / std) ** 3))
    kurt = float(np.mean(((seg - mean) / std) ** 4))
    fft  = np.abs(np.fft.rfft(seg)) ** 2
    tp   = fft.sum() + 1e-10
    lf   = fft[:len(fft) // 4].sum() / tp
    hf   = fft[len(fft) // 4:].sum() / tp
    base = np.array([mean, std, rms, ptp, skew, kurt, lf, hf], dtype=float)

    extra = np.zeros(4, dtype=float)

    if sig_type in ("ECG", "BVP", "PPG"):
        hrv = _hrv_features(seg, fs)
        extra = np.array([hrv[3], hrv[2], hrv[7], hrv[4]])
        # [Mean BPM, RMSSD, LF/HF, pNN50]

    elif sig_type == "RESP":
        try:
            filt    = _bandpass(seg, fs, 0.05, 2.0)
            crosses = np.where(np.diff(np.sign(filt)))[0]
            rate    = (len(crosses) / 2) / (len(seg) / fs) * 60
            amp     = np.ptp(filt)
            irreg   = np.std(np.diff(crosses)) if len(crosses) > 2 else 0.0
            extra[:3] = [rate, amp, irreg]
        except Exception:
            pass

    elif sig_type in ("EDA", "GSR"):
        try:
            b, a = butter(2, min(0.05 / (fs / 2), 0.99), btype="low")
            scl  = filtfilt(b, a, seg)
            scr  = seg - scl
            n_pk, _ = find_peaks(scr, height=0.02 * std, distance=int(fs))
            extra[:3] = [np.mean(scl), len(n_pk), np.std(scr)]
        except Exception:
            pass

    elif sig_type == "EMG":
        try:
            hi = _bandpass(seg, fs, 20.0, min(450.0, fs * 0.45))
            extra[0] = np.sqrt(np.mean(hi ** 2))
            extra[1] = np.mean(np.abs(np.diff(seg)))
        except Exception:
            pass

    elif sig_type in ("SKT", "TEMP"):
        extra[0] = mean
        extra[1] = seg[-1] - seg[0]
        extra[2] = std

    return np.concatenate([base, extra])


FEATURE_DIM   = 12
FEATURE_NAMES = [
    "Mean", "Std", "RMS", "PtP", "Skewness", "Kurtosis",
    "LF power", "HF power",
    "Physio 1", "Physio 2", "Physio 3", "Physio 4",
]

PHYSIO_LABELS: dict[str, list[str]] = {
    "ECG":  ["Mean BPM", "RMSSD", "LF/HF ratio", "pNN50"],
    "BVP":  ["Mean BPM", "RMSSD", "LF/HF ratio", "pNN50"],
    "PPG":  ["Mean BPM", "RMSSD", "LF/HF ratio", "pNN50"],
    "RESP": ["Resp rate (brpm)", "Amplitude", "Irregularity", "—"],
    "EDA":  ["SCL mean", "SCR peaks", "SCR std", "—"],
    "GSR":  ["SCL mean", "SCR peaks", "SCR std", "—"],
    "EMG":  ["EMG RMS", "MAV deriv.", "—", "—"],
    "SKT":  ["Skin temp", "Temp trend", "Temp std", "—"],
    "TEMP": ["Skin temp", "Temp trend", "Temp std", "—"],
}


def extract_features_windowed(
    signal: np.ndarray,
    fs: float,
    sig_type: str = "",
    window_sec: float = 10.0,
    step_sec:   float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    sig    = signal.ravel().astype(float)
    win_s  = int(window_sec * fs)
    step_s = max(1, int(step_sec * fs))

    features, centers = [], []
    start = 0
    while start + win_s <= len(sig):
        features.append(_window_features(sig[start: start + win_s], fs, sig_type))
        centers.append((start + win_s / 2) / fs)
        start += step_s

    if not features:
        return np.empty((0, FEATURE_DIM)), np.array([])
    return np.array(features, dtype=float), np.array(centers)


# ── Physiological stress labels ───────────────────────────────────────────────

def _physio_score(X: np.ndarray, sig_name: str) -> np.ndarray:
    """Composite stress score for one signal. Higher = more stressed."""
    def _norm(a: np.ndarray) -> np.ndarray:
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 1e-10 else np.zeros_like(a)

    if sig_name in ("ECG", "BVP", "PPG"):
        # High BPM + low RMSSD + high LF/HF = stress
        return _norm(X[:, 8]) + _norm(-X[:, 9]) + _norm(X[:, 10])
    elif sig_name == "RESP":
        return _norm(X[:, 8])   # respiratory rate
    elif sig_name in ("EDA", "GSR"):
        return _norm(X[:, 8])   # SCL mean
    elif sig_name == "EMG":
        return _norm(X[:, 8])   # EMG RMS
    elif sig_name in ("SKT", "TEMP"):
        return _norm(-X[:, 8])  # temperature drop = stress
    else:
        return _norm(X[:, 2])   # generic RMS


def build_physiological_labels(
    signals: dict[str, np.ndarray],
    fs_map:  dict[str, float],
    window_sec: float = 10.0,
    step_sec:   float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted majority vote across all signals.
    ECG/cardiac signals vote twice. Upper tertile = stress.
    Returns (t_centers, y_stress).
    """
    WEIGHTS = {"ECG": 2, "BVP": 2, "PPG": 2,
               "EDA": 1, "GSR": 1, "RESP": 1, "EMG": 1, "SKT": 1, "TEMP": 1}

    weighted_votes: dict[int, list] = {}
    common_t: np.ndarray | None = None
    n_win = 0

    for sig_name, signal in signals.items():
        fs  = fs_map.get(sig_name, 2000)
        X, t = extract_features_windowed(signal, fs, sig_name, window_sec, step_sec)
        if len(X) == 0:
            continue

        score  = _physio_score(X, sig_name)
        thresh = np.percentile(score, 67)   # upper third = stress
        vote   = (score >= thresh).astype(int)
        weight = WEIGHTS.get(sig_name, 1)

        if common_t is None:
            common_t = t
            n_win    = len(t)
        else:
            n_win = min(n_win, len(t))

        for i in range(min(len(vote), n_win)):
            weighted_votes.setdefault(i, []).extend([int(vote[i])] * weight)

    if common_t is None or not weighted_votes:
        return np.array([]), np.array([], dtype=int)

    t_out = common_t[:n_win]
    y_out = np.array([
        1 if sum(weighted_votes.get(i, [0])) > len(weighted_votes.get(i, [1])) / 2
        else 0
        for i in range(n_win)
    ], dtype=int)

    return t_out, y_out


# ── Session-interval labels (ROC / boxplots only) ─────────────────────────────

def build_labels_from_intervals(
    t_centers: np.ndarray,
    stress_intervals: list[tuple[float, float]],
) -> np.ndarray:
    labels = np.zeros(len(t_centers), dtype=int)
    for start, end in stress_intervals:
        labels[(t_centers >= start) & (t_centers <= end)] = 1
    return labels

build_labels = build_labels_from_intervals


# ── Stress Model (Gradient Boosting > RF on short physiological windows) ──────

class StressModel:
    """GradientBoosting pipeline. Falls back to RandomForest if needed."""

    def __init__(self, random_state: int = 42):
        self._rs      = random_state
        self.pipeline: Pipeline | None = None
        self.trained  = False

    def _make(self, use_gbm: bool) -> Pipeline:
        clf = (
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=self._rs,
            ) if use_gbm else
            RandomForestClassifier(
                n_estimators=300, max_depth=6,
                class_weight="balanced",
                n_jobs=-1, random_state=self._rs,
            )
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            self.pipeline = self._make(use_gbm=True)
            self.pipeline.fit(X, y)
        except Exception:
            self.pipeline = self._make(use_gbm=False)
            self.pipeline.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def feature_importances(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].feature_importances_

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "StressModel":
        obj = cls()
        obj.pipeline = joblib.load(path)
        obj.trained  = True
        return obj