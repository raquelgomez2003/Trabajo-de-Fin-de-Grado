"""
models.py
Feature extraction, stress classification, ECG BPM and RR interval analysis.

Pipeline:
  - Beat detection via bandpass + peak finding
  - RR intervals computed beat-by-beat
  - HRV features: RMSSD, SDNN, pNN50, LF/HF ratio
  - RandomForest classifier (fast, good enough for small datasets)
  - Physiological stress labelling with weighted majority vote

Empatica-specific stress estimation
------------------------------------
Las senales propias de Empatica (BVP, HR, TEMP, ACC) tienen ahora su PROPIA
extraccion de caracteristicas y su propio 'physio score', adaptados a lo que
cada senal mide realmente. Las senales de Biopac (ECG, EDA, EMG, PPG, RESP,
SKT) NO se han tocado: conservan exactamente el mismo codigo que antes.

Public API
----------
compute_rr(ecg, fs)                 -> (rr_times, rr_ms)
compute_bpm(ecg, fs, ...)           -> (t_bpm, bpm_values)
extract_features_windowed(...)      -> (X, t_centers)
build_physiological_labels(...)     -> (t_centers, y)
build_labels_from_intervals(...)    -> y
class StressModel
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
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
    std = np.std(filtered)
    if std < 1e-10:
        return np.array([], dtype=int)
    # Altura relativa más permisiva para señales en µV
    height = 0.1 * std   # ← era 0.3, bajamos a 0.1
    peaks, _ = find_peaks(filtered,
                          distance=int(min_rr_sec * fs),
                          height=height)
    return peaks
# ── RR intervals ──────────────────────────────────────────────────────────────

def compute_rr(ecg_signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Beat-by-beat RR intervals. Returns (rr_times, rr_ms)."""
    peaks = _detect_peaks(ecg_signal.ravel(), fs)
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
    ecg_1d       = ecg_signal.ravel()
    peaks        = _detect_peaks(ecg_1d, fs, min_rr_sec)
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


# ── HRV features ──────────────────────────────────────────────────────────────

def _hrv_features(segment: np.ndarray, fs: float) -> np.ndarray:
    """8 HRV features. Returns zeros if not enough peaks."""
    feats = np.zeros(8, dtype=float)
    try:
        peaks = _detect_peaks(segment, fs)
        if len(peaks) < 3:
            return feats
        rr      = np.diff(peaks) / fs * 1000.0  # ms
        diff_rr = np.diff(rr)

        feats[0] = np.mean(rr)                                          # mean RR
        feats[1] = np.std(rr)                                           # SDNN
        feats[2] = np.sqrt(np.mean(diff_rr ** 2))                       # RMSSD
        feats[3] = 60000.0 / (np.mean(rr) + 1e-10)                      # mean BPM
        feats[4] = np.sum(np.abs(diff_rr) > 50) / max(len(diff_rr), 1) # pNN50

        if len(rr) >= 8:
            freqs    = np.fft.rfftfreq(len(rr), d=np.mean(rr) / 1000.0)
            power    = np.abs(np.fft.rfft(rr - rr.mean())) ** 2
            lf_p     = power[(freqs >= 0.04) & (freqs < 0.15)].sum()
            hf_p     = power[(freqs >= 0.15) & (freqs < 0.40)].sum() + 1e-10
            feats[5] = lf_p
            feats[6] = hf_p
            feats[7] = lf_p / hf_p  # LF/HF — key stress marker
    except Exception:
        pass
    return feats


HRV_FEATURE_NAMES = [
    "Mean RR (ms)", "SDNN (ms)", "RMSSD (ms)", "Mean BPM",
    "pNN50", "LF power", "HF power", "LF/HF ratio",
]


# ── Empatica-specific feature extractors ──────────────────────────────────────
# Cada uno devuelve EXACTAMENTE 4 valores (los 4 "physio" slots de FEATURE_DIM).
# Solo se usan con señales que Biopac no tiene, así que no afectan a Biopac.

def _bvp_features(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    HRV a partir de la onda de pulso BVP de Empatica (~64 Hz).
    A diferencia del ECG (QRS), el BVP es una onda de pulso suave: se filtra en
    banda de pulso (0.5–8 Hz) y se detectan los picos sistólicos.
    Devuelve [BPM medio, RMSSD, LF/HF, pNN50].
    """
    feats = np.zeros(4, dtype=float)
    try:
        filt = _bandpass(seg, fs, 0.5, min(8.0, fs * 0.45))
        s = np.std(filt)
        if s < 1e-10:
            return feats
        peaks, _ = find_peaks(filt,
                              distance=max(1, int(0.4 * fs)),  # <=150 BPM
                              height=0.2 * s)
        if len(peaks) < 3:
            return feats
        rr = np.diff(peaks) / fs * 1000.0            # ms
        rr = rr[(rr >= 300) & (rr <= 2000)]          # 30–200 BPM
        if len(rr) < 3:
            return feats
        drr = np.diff(rr)
        feats[0] = 60000.0 / np.mean(rr)                        # BPM medio
        feats[1] = np.sqrt(np.mean(drr ** 2))                   # RMSSD
        feats[3] = np.sum(np.abs(drr) > 50) / max(len(drr), 1)  # pNN50
        if len(rr) >= 8:
            freqs = np.fft.rfftfreq(len(rr), d=np.mean(rr) / 1000.0)
            power = np.abs(np.fft.rfft(rr - rr.mean())) ** 2
            lf = power[(freqs >= 0.04) & (freqs < 0.15)].sum()
            hf = power[(freqs >= 0.15) & (freqs < 0.40)].sum() + 1e-10
            feats[2] = lf / hf                                  # LF/HF
    except Exception:
        pass
    return feats


def _hr_features(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    Caracteristicas de la señal HR de Empatica (ya viene en BPM, ~1 Hz).
    Devuelve [HR media, variabilidad corta (media|ΔHR|), tendencia (pendiente),
    fraccion de tiempo por encima de la basal].
    Mas HR, tendencia positiva y menos variabilidad -> mas estres.
    """
    feats = np.zeros(4, dtype=float)
    hr = np.asarray(seg, dtype=float).ravel()
    hr = hr[(hr > 20) & (hr < 220)]      # descartar valores imposibles
    if len(hr) < 3:
        if len(hr):
            feats[0] = float(np.mean(hr))
        return feats
    feats[0] = float(np.mean(hr))                       # HR media
    feats[1] = float(np.mean(np.abs(np.diff(hr))))      # variabilidad corta
    x = np.arange(len(hr))
    feats[2] = float(np.polyfit(x, hr, 1)[0])           # tendencia (BPM/muestra)
    baseline = np.percentile(hr, 25)                    # nivel de reposo
    feats[3] = float(np.mean(hr > baseline))            # fraccion elevada
    return feats


def _temp_features(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    Temperatura de piel (Empatica TEMP, ~4 Hz).
    Devuelve [temp media, deriva (fin-inicio), std, pendiente].
    El estres produce vasoconstriccion periferica -> la temperatura BAJA
    (deriva/pendiente negativa).
    """
    feats = np.zeros(4, dtype=float)
    t = np.asarray(seg, dtype=float).ravel()
    if len(t) < 2:
        if len(t):
            feats[0] = float(np.mean(t))
        return feats
    feats[0] = float(np.mean(t))
    feats[1] = float(t[-1] - t[0])
    feats[2] = float(np.std(t))
    x = np.arange(len(t))
    feats[3] = float(np.polyfit(x, t, 1)[0])
    return feats


def _acc_features(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    Movimiento (magnitud del acelerometro Empatica, ~32 Hz).
    Devuelve [magnitud media, std, nivel de actividad (media|Δacc|), rango].
    Sirve sobre todo como contexto/confusor: mucho movimiento no es estres.
    """
    feats = np.zeros(4, dtype=float)
    a = np.asarray(seg, dtype=float).ravel()
    if len(a) < 2:
        return feats
    feats[0] = float(np.mean(a))
    feats[1] = float(np.std(a))
    feats[2] = float(np.mean(np.abs(np.diff(a))))
    feats[3] = float(np.ptp(a))
    return feats


# ── Per-window features ───────────────────────────────────────────────────────

def _window_features(segment: np.ndarray, fs: float, sig_type: str) -> np.ndarray:
    seg  = segment.astype(float)
    mean = np.mean(seg)
    std  = np.std(seg) + 1e-10
    fft  = np.abs(np.fft.rfft(seg)) ** 2
    tp   = fft.sum() + 1e-10
    base = np.array([
        mean, std,
        np.sqrt(np.mean(seg ** 2)),         # RMS
        np.ptp(seg),                         # peak-to-peak
        float(np.mean(((seg - mean) / std) ** 3)),  # skewness
        float(np.mean(((seg - mean) / std) ** 4)),  # kurtosis
        fft[:len(fft) // 4].sum() / tp,     # LF power
        fft[len(fft) // 4:].sum() / tp,     # HF power
    ], dtype=float)

    extra = np.zeros(4, dtype=float)

    # ── Biopac (SIN CAMBIOS) ──────────────────────────────────────────────────
    if sig_type in ("ECG", "PPG"):
        hrv   = _hrv_features(seg, fs)
        extra = np.array([hrv[3], hrv[2], hrv[7], hrv[4]])  # BPM, RMSSD, LF/HF, pNN50

    elif sig_type == "RESP":
        try:
            filt    = _bandpass(seg, fs, 0.05, 2.0)
            crosses = np.where(np.diff(np.sign(filt)))[0]
            extra[:3] = [
                (len(crosses) / 2) / (len(seg) / fs) * 60,  # resp rate
                np.ptp(filt),                                 # amplitude
                np.std(np.diff(crosses)) if len(crosses) > 2 else 0.0,  # irregularity
            ]
        except Exception:
            pass

    elif sig_type in ("EDA", "GSR"):
        try:
            b, a  = butter(2, min(0.05 / (fs / 2), 0.99), btype="low")
            scl   = filtfilt(b, a, seg)
            scr   = seg - scl
            n_pk, _ = find_peaks(scr, height=0.02 * std, distance=int(fs))
            extra[:3] = [np.mean(scl), len(n_pk), np.std(scr)]
        except Exception:
            pass

    elif sig_type == "EMG":
        try:
            hi = _bandpass(seg, fs, 20.0, min(450.0, fs * 0.45))
            extra[:2] = [np.sqrt(np.mean(hi ** 2)), np.mean(np.abs(np.diff(seg)))]
        except Exception:
            pass

    elif sig_type == "SKT":
        extra[:3] = [mean, seg[-1] - seg[0], std]

    # ── Empatica (NUEVO / mejorado) ───────────────────────────────────────────
    elif sig_type == "BVP":
        extra = _bvp_features(seg, fs)

    elif sig_type == "HR":
        extra = _hr_features(seg, fs)

    elif sig_type == "TEMP":
        extra = _temp_features(seg, fs)

    elif sig_type == "ACC":
        extra = _acc_features(seg, fs)

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
    "HR":   ["Mean HR", "HR var (|ΔHR|)", "HR trend", "% above baseline"],
    "RESP": ["Resp rate (brpm)", "Amplitude", "Irregularity", "—"],
    "EDA":  ["SCL mean", "SCR peaks", "SCR std", "—"],
    "GSR":  ["SCL mean", "SCR peaks", "SCR std", "—"],
    "EMG":  ["EMG RMS", "MAV deriv.", "—", "—"],
    "SKT":  ["Skin temp", "Temp trend", "Temp std", "—"],
    "TEMP": ["Skin temp", "Temp drift", "Temp std", "Temp slope"],
    "ACC":  ["Mean |acc|", "Acc std", "Activity", "Range"],
}


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_windowed(
    signal: np.ndarray,
    fs: float,
    sig_type: str = "",
    window_sec: float = 30.0,   # increased from 10s → faster
    step_sec:   float = 15.0,   # increased from 5s  → faster
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


def extract_features_all_signals(
    signals: dict[str, np.ndarray],
    fs_map:  dict[str, float],
    window_sec: float = 30.0,
    step_sec:   float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae features de todas las señales y las concatena en una sola matriz.
    Cada fila = una ventana de tiempo, columnas = features de todas las señales.
    Returns (X, t_centers).
    """
    all_X:  list[np.ndarray] = []
    common_t: np.ndarray | None = None
    n_win = None

    for sig_name, signal in signals.items():
        fs = fs_map.get(sig_name, 2000)
        X, t = extract_features_windowed(signal, fs, sig_name, window_sec, step_sec)
        if len(X) == 0:
            continue
        if common_t is None:
            common_t = t
            n_win    = len(t)
        else:
            n_win = min(n_win, len(t))
        all_X.append(X[:n_win])

    if not all_X or common_t is None:
        return np.empty((0, 0)), np.array([])

    # Recortar todas al mismo número de ventanas y concatenar columnas
    n_win    = min(len(X) for X in all_X)
    X_concat = np.concatenate([X[:n_win] for X in all_X], axis=1)
    return X_concat, common_t[:n_win]

# ── Physiological stress labels ───────────────────────────────────────────────

def _physio_score(X: np.ndarray, sig_name: str) -> np.ndarray:
    """Composite stress score for one signal. Higher = more stressed."""
    def _norm(a: np.ndarray) -> np.ndarray:
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 1e-10 else np.zeros_like(a)

    # ── Biopac (SIN CAMBIOS) ──────────────────────────────────────────────────
    if sig_name in ("ECG", "PPG"):
        return _norm(X[:, 8]) + _norm(-X[:, 9]) + _norm(X[:, 10])
    elif sig_name == "RESP":
        return _norm(X[:, 8])
    elif sig_name in ("EDA", "GSR"):
        return _norm(X[:, 8])
    elif sig_name == "EMG":
        return _norm(X[:, 8])
    elif sig_name == "SKT":
        return _norm(-X[:, 8])

    # ── Empatica (mejorado) ───────────────────────────────────────────────────
    elif sig_name == "BVP":
        # BPM alto + RMSSD bajo + LF/HF alto -> estres
        return _norm(X[:, 8]) + _norm(-X[:, 9]) + _norm(X[:, 10])
    elif sig_name == "HR":
        # HR media alta + tendencia creciente -> estres
        return _norm(X[:, 8]) + _norm(X[:, 10])
    elif sig_name == "TEMP":
        # temperatura media baja + pendiente negativa -> estres
        return _norm(-X[:, 8]) + _norm(-X[:, 11])
    elif sig_name == "ACC":
        # movimiento: señal debil, solo como apoyo
        return _norm(X[:, 8])
    else:
        return _norm(X[:, 2])


def build_physiological_labels(
    signals: dict[str, np.ndarray],
    fs_map:  dict[str, float],
    window_sec: float = 30.0,
    step_sec:   float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted majority vote across all signals.
    ECG/cardiac signals vote twice. Upper tertile = stress.
    Returns (t_centers, y_stress).
    """
    WEIGHTS = {"ECG": 2, "BVP": 2, "PPG": 2, "HR": 2,
               "EDA": 1, "GSR": 1, "RESP": 1, "EMG": 1,
               "SKT": 1, "TEMP": 1, "ACC": 0}

    all_votes: list[np.ndarray] = []
    all_weights: list[int]      = []
    common_t: np.ndarray | None = None

    for sig_name, signal in signals.items():
        fs   = fs_map.get(sig_name, 2000)
        X, t = extract_features_windowed(signal, fs, sig_name, window_sec, step_sec)
        if len(X) == 0:
            continue

        score  = _physio_score(X, sig_name)
        vote   = (score >= np.percentile(score, 67)).astype(int)
        n_win  = len(t) if common_t is None else min(len(common_t), len(t))

        if common_t is None:
            common_t = t

        all_votes.append(vote[:n_win])
        all_weights.append(WEIGHTS.get(sig_name, 1))

    if common_t is None or not all_votes:
        return np.array([]), np.array([], dtype=int)

    n_win    = min(len(v) for v in all_votes)
    common_t = common_t[:n_win]

    # Weighted sum across signals — majority = stress
    weighted_sum   = sum(v[:n_win] * w for v, w in zip(all_votes, all_weights))
    total_weight   = sum(all_weights) if sum(all_weights) > 0 else 1

    y_out          = (weighted_sum > total_weight / 2).astype(int)

    return common_t, y_out


# ── Interval-based labels ─────────────────────────────────────────────────────

def build_labels_from_intervals(
    t_centers: np.ndarray,
    stress_intervals: list[tuple[float, float]],
) -> np.ndarray:
    labels = np.zeros(len(t_centers), dtype=int)
    for start, end in stress_intervals:
        labels[(t_centers >= start) & (t_centers <= end)] = 1
    return labels

build_labels = build_labels_from_intervals


# ── Stress Model ──────────────────────────────────────────────────────────────

class StressModel:
    """RandomForest classifier — fast and reliable for small physiological datasets."""

    def __init__(self, random_state: int = 42):
        self._rs      = random_state
        self.pipeline: Pipeline | None = None
        self.trained  = False

    def _make(self) -> Pipeline:
        clf = RandomForestClassifier(
            n_estimators=50,    # reduced from 200/300 → much faster
            max_depth=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self._rs,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.pipeline = self._make()
        self.pipeline.fit(X, y)
        self.trained  = True

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
        obj          = cls()
        obj.pipeline = joblib.load(path)
        obj.trained  = True
        return obj