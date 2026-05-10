"""
models.py
Feature extraction, physiological stress detection (Random Forest), and ECG BPM.

Stress is detected from the signals themselves using physiological markers:
  - ECG/BVP : elevated BPM (heart rate)
  - RESP     : increased respiratory rate and amplitude
  - EDA      : elevated skin conductance (arousal)
  - EMG      : increased muscle tension
  - SKT/TEMP : skin temperature drop (sympathetic response)
  - PPG      : reduced pulse amplitude, higher rate

Labels are built by combining ALL available signals into a composite
physiological stress score, then thresholding — not from session intervals.
Session intervals (calming/vexing) are only used for the boxplot/ROC comparison.

Public API
----------
extract_features_windowed(signal, fs, sig_type, window_sec, step_sec)
    → (features: np.ndarray, t_centers: np.ndarray)

build_physiological_labels(signals, fs_map, window_sec, step_sec)
    → (t_centers, y_stress) — labels derived from signal physiology

train_stress_model(X, y) → StressModel

compute_bpm(ecg_signal, fs) → (bpm_times, bpm_values)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


# ── Signal-specific feature extraction ────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: float,
              low: float = 0.5, high: float = 40.0) -> np.ndarray:
    nyq = fs / 2.0
    low_n  = max(low  / nyq, 1e-4)
    high_n = min(high / nyq, 0.9999)
    b, a = butter(4, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def _window_features(
    segment: np.ndarray,
    fs: float,
    sig_type: str,
) -> np.ndarray:
    """
    Extract physiologically meaningful features per signal type.
    All feature vectors are padded to FEATURE_DIM entries.
    """
    seg = segment.astype(float)
    n   = len(seg)

    # ── Universal statistical features (8) ────────────────────────────────
    mean  = np.mean(seg)
    std   = np.std(seg) + 1e-10
    rms   = np.sqrt(np.mean(seg ** 2))
    ptp   = np.ptp(seg)
    skew  = float(np.mean(((seg - mean) / std) ** 3))
    kurt  = float(np.mean(((seg - mean) / std) ** 4))
    fft   = np.abs(np.fft.rfft(seg)) ** 2
    tp    = fft.sum() + 1e-10
    lf    = fft[:len(fft) // 4].sum() / tp
    hf    = fft[len(fft) // 4:].sum() / tp
    base  = np.array([mean, std, rms, ptp, skew, kurt, lf, hf], dtype=float)

    # ── Signal-specific physiological features (4) ────────────────────────
    extra = np.zeros(4, dtype=float)

    if sig_type in ("ECG", "BVP", "PPG"):
        # Heart rate variability proxy: std of RR intervals
        try:
            filtered  = _bandpass(seg, fs, 0.5, 40.0)
            min_dist  = int(0.3 * fs)
            peaks, _  = find_peaks(filtered, distance=min_dist,
                                   height=0.3 * np.std(filtered))
            if len(peaks) >= 2:
                rr        = np.diff(peaks) / fs          # RR in seconds
                bpm_mean  = 60.0 / rr.mean()
                bpm_std   = 60.0 * rr.std() / (rr.mean() ** 2 + 1e-10)
                hrv_rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
                extra[:3] = [bpm_mean, bpm_std, hrv_rmssd]
                # HF/LF power ratio (sympathetic/parasympathetic balance)
                freqs = np.fft.rfftfreq(n, 1 / fs)
                lf_p  = fft[(freqs >= 0.04) & (freqs < 0.15)].sum()
                hf_p  = fft[(freqs >= 0.15) & (freqs < 0.4)].sum() + 1e-10
                extra[3] = lf_p / hf_p   # high LF/HF → sympathetic dominance → stress
        except Exception:
            pass

    elif sig_type == "RESP":
        # Respiratory rate from zero crossings + amplitude
        try:
            filtered    = _bandpass(seg, fs, 0.1, 1.0)
            crossings   = np.where(np.diff(np.sign(filtered)))[0]
            if len(crossings) >= 2:
                resp_rate = (len(crossings) / 2) / (n / fs) * 60  # breaths/min
            else:
                resp_rate = 0.0
            resp_amp    = np.ptp(filtered)
            resp_irreg  = np.std(np.diff(crossings)) if len(crossings) > 2 else 0.0
            extra[:3]   = [resp_rate, resp_amp, resp_irreg]
        except Exception:
            pass

    elif sig_type in ("EDA", "GSR"):
        # Skin conductance level + number of SCR peaks
        try:
            scl      = np.mean(seg)
            scr      = seg - filtfilt(*butter(2, 0.05 / (fs / 2), btype="low"), seg)
            n_peaks, _ = find_peaks(scr, height=0.02 * np.std(scr),
                                    distance=int(fs))
            extra[:3] = [scl, len(n_peaks), np.std(scr)]
        except Exception:
            pass

    elif sig_type == "EMG":
        # Muscle activation: RMS of high-frequency content
        try:
            hi = _bandpass(seg, fs, 20.0, min(500.0, fs * 0.45))
            extra[0] = np.sqrt(np.mean(hi ** 2))   # EMG RMS (activation)
            extra[1] = np.mean(np.abs(np.diff(seg)))  # mean absolute value derivative
        except Exception:
            pass

    elif sig_type in ("SKT", "TEMP"):
        # Skin temp drop → sympathetic arousal (stress causes vasoconstriction)
        extra[0] = mean
        extra[1] = seg[-1] - seg[0]   # temperature trend (negative = drop = stress)
        extra[2] = std

    return np.concatenate([base, extra])


FEATURE_DIM   = 12   # 8 statistical + 4 physiological
FEATURE_NAMES = [
    "Mean", "Std", "RMS", "PtP", "Skewness", "Kurtosis",
    "LF power", "HF power",
    "Physio 1", "Physio 2", "Physio 3", "Physio 4",
]

# Per-signal human-readable names for the physio features
PHYSIO_LABELS: dict[str, list[str]] = {
    "ECG":  ["BPM mean", "BPM std", "HRV RMSSD", "LF/HF ratio"],
    "BVP":  ["BPM mean", "BPM std", "HRV RMSSD", "LF/HF ratio"],
    "PPG":  ["BPM mean", "BPM std", "HRV RMSSD", "LF/HF ratio"],
    "RESP": ["Resp rate (brpm)", "Resp amplitude", "Irregularity", "—"],
    "EDA":  ["SCL mean", "SCR peaks", "SCR std", "—"],
    "GSR":  ["SCL mean", "SCR peaks", "SCR std", "—"],
    "EMG":  ["EMG RMS", "MAV derivative", "—", "—"],
    "SKT":  ["Skin temp", "Temp trend", "Temp std", "—"],
    "TEMP": ["Skin temp", "Temp trend", "Temp std", "—"],
}


def extract_features_windowed(
    signal: np.ndarray,
    fs: float,
    sig_type: str = "",
    window_sec: float = 60.0,
    step_sec: float   = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over *signal* and extract features per window.
    Returns (X [n_windows, FEATURE_DIM], t_centers [n_windows]) in seconds.
    """
    if signal.ndim == 2:
        signal = signal.ravel() if signal.shape[1] == 1 else signal[:, 0]

    win_samples  = int(window_sec * fs)
    step_samples = max(1, int(step_sec * fs))

    features, centers = [], []
    start = 0
    while start + win_samples <= len(signal):
        seg = signal[start: start + win_samples]
        features.append(_window_features(seg, fs, sig_type))
        centers.append((start + win_samples / 2) / fs)
        start += step_samples

    if not features:
        return np.empty((0, FEATURE_DIM)), np.array([])

    return np.array(features, dtype=float), np.array(centers)


# ── Physiological stress labelling ────────────────────────────────────────────

def build_physiological_labels(
    signals: dict[str, np.ndarray],
    fs_map:  dict[str, float],
    window_sec: float = 60.0,
    step_sec:   float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive stress labels purely from the physiological signals.

    Strategy (per window, per available signal):
      - ECG/BVP/PPG : BPM > mean + 0.5·std  → stress indicator
      - RESP         : resp rate > mean + 0.5·std OR amplitude > threshold
      - EDA          : SCL > mean + 0.5·std  → stress indicator
      - EMG          : RMS > mean + 0.5·std  → stress indicator
      - SKT/TEMP     : temp < mean - 0.5·std → stress (vasoconstriction)

    Each signal votes; a window is labelled stress=1 if the majority vote.
    Returns (t_centers, y_labels) — t_centers is from the shortest signal.
    """
    votes_per_window: dict[int, list[int]] = {}
    common_t: np.ndarray | None = None

    for sig_name, signal in signals.items():
        fs  = fs_map.get(sig_name, 2000)
        X, t = extract_features_windowed(signal, fs, sig_name, window_sec, step_sec)
        if len(X) == 0:
            continue

        # Build a stress score for this signal using its key physio feature
        stress_score = _physio_stress_score(X, sig_name)

        # Threshold: above/below signal mean ± 0.5 std
        thresh = stress_score.mean() + 0.5 * stress_score.std()

        if sig_name in ("SKT", "TEMP"):
            # Lower temperature = more stress
            vote = (stress_score < stress_score.mean() - 0.5 * stress_score.std()).astype(int)
        else:
            vote = (stress_score > thresh).astype(int)

        # Align on time axis (use shortest common length)
        if common_t is None:
            common_t = t
            n_win    = len(t)
        else:
            n_win = min(n_win, len(t))

        for i in range(min(len(vote), n_win)):
            votes_per_window.setdefault(i, []).append(int(vote[i]))

    if common_t is None or not votes_per_window:
        return np.array([]), np.array([], dtype=int)

    t_out = common_t[:n_win]
    y_out = np.array([
        1 if sum(votes_per_window.get(i, [0])) > len(votes_per_window.get(i, [0])) / 2
        else 0
        for i in range(n_win)
    ], dtype=int)

    return t_out, y_out


def _physio_stress_score(X: np.ndarray, sig_name: str) -> np.ndarray:
    """Extract the single most relevant physiological feature for stress voting."""
    if sig_name in ("ECG", "BVP", "PPG"):
        return X[:, 8]   # BPM mean (physio feature 1)
    elif sig_name == "RESP":
        return X[:, 8]   # Respiratory rate
    elif sig_name in ("EDA", "GSR"):
        return X[:, 8]   # SCL mean
    elif sig_name == "EMG":
        return X[:, 8]   # EMG RMS
    elif sig_name in ("SKT", "TEMP"):
        return X[:, 8]   # Skin temperature (inverted logic in caller)
    else:
        return X[:, 2]   # RMS as generic fallback


# ── Session-interval label builder (for ROC/boxplot comparison only) ──────────

def build_labels_from_intervals(
    t_centers: np.ndarray,
    stress_intervals: list[tuple[float, float]],
) -> np.ndarray:
    """
    Label windows by session interval (calming/vexing).
    Used ONLY for ROC and boxplot phase comparison — NOT for RF training.
    """
    labels = np.zeros(len(t_centers), dtype=int)
    for start, end in stress_intervals:
        labels[(t_centers >= start) & (t_centers <= end)] = 1
    return labels

# Keep old name as alias for backwards compatibility
build_labels = build_labels_from_intervals


# ── Stress Model ──────────────────────────────────────────────────────────────

class StressModel:
    """RandomForest wrapper with integrated scaler."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.scaler  = StandardScaler()
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
        obj  = cls()
        data = joblib.load(path)
        obj.clf    = data["clf"]
        obj.scaler = data["scaler"]
        obj.trained = True
        return obj


# ── BPM from ECG ─────────────────────────────────────────────────────────────

def compute_bpm(
    ecg_signal: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    step_sec:   float = 5.0,
    min_rr_sec: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks and compute sliding-window BPM.
    Returns (t_bpm [seconds], bpm_values).
    Signal is always flattened to 1D regardless of input shape.
    """
    ecg_1d = ecg_signal.ravel()   # works for both 1D and any 2D shape

    try:
        filtered = _bandpass(ecg_1d, fs, 0.5, 40.0)
    except Exception:
        return np.array([]), np.array([])

    min_dist = int(min_rr_sec * fs)
    height   = 0.3 * np.std(filtered)
    peaks, _ = find_peaks(filtered, distance=min_dist, height=height)

    if len(peaks) < 2:
        return np.array([]), np.array([])

    peak_times   = peaks / fs
    win_samples  = int(window_sec * fs)
    step_samples = max(1, int(step_sec * fs))

    bpm_times, bpm_values = [], []
    start = 0
    while start + win_samples <= len(ecg_1d):
        t_start  = start / fs
        t_end    = (start + win_samples) / fs
        rr_peaks = peak_times[(peak_times >= t_start) & (peak_times < t_end)]
        if len(rr_peaks) >= 2:
            mean_rr = np.diff(rr_peaks).mean()
            bpm_values.append(60.0 / mean_rr)
            bpm_times.append((t_start + t_end) / 2)
        start += step_samples

    return np.array(bpm_times), np.array(bpm_values)