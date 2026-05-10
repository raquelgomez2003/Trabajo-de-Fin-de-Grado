"""
config.py
Global configuration — BioVisor App.
All constants, device presets, and signal metadata live here.
"""

# ── Device frequency presets ───────────────────────────────────────────────────
DEVICE_FS = {
    "Biopac": {
        "ECG":  2000,
        "EDA":  2000,
        "EMG":  2000,
        "PPG":  2000,
        "RESP": 2000,
        "SKT":  2000,
    },
    "Empatica": {
        # Empatica E4 standard rates — update if your firmware differs
        "BVP":  64,
        "EDA":  4,
        "TEMP": 4,
        "ACC":  32,
        "HR":   1,
    },
}

# Signal types recognised per device
DEVICE_SIGNALS = {
    "Biopac":   ["ECG", "EDA", "EMG", "PPG", "RESP", "SKT"],
    "Empatica": ["BVP", "EDA", "TEMP", "ACC", "HR"],
}

# Colours used for stress phases in plots
PHASE_COLORS = {
    "baseline": ("#d0e8ff", "Baseline"),
    "calming":  ("#b6f0c8", "Calming music"),
    "relax":    ("#c8d8ff", "Relax"),
    "vexing":   ("#ffb3b3", "Vexing / Stress"),
}

# ECG 12-lead column names (Biopac export)
ECG_COLUMNS = ["I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6"]

# Random Forest feature window (seconds)
RF_WINDOW_SEC = 60      # 1-minute sliding windows
RF_STEP_SEC   = 30      # 50 % overlap

# BPM detection: minimum peak distance (seconds)
ECG_MIN_RR_SEC = 0.3    # ~200 BPM upper bound

# ── Artifact / outlier filter parameters per signal type ──────────────────────
#
# Each entry defines how clean_signal() in filters.py treats that signal.
#
#   iqr_factor   : values outside median ± iqr_factor·IQR are clipped → interpolated.
#                  Lower = more aggressive. Typical range 2–6.
#                  Use None to disable outlier removal for that signal.
#
#   bandpass     : (low_hz, high_hz) physiological band to keep, or None to skip.
#                  Applied after outlier removal using a zero-phase Butterworth filter.
#
#   savgol       : (window_samples, poly_order) Savitzky-Golay smooth, or None to skip.
#                  Good for slow signals (EDA, SKT, TEMP) that should stay smooth.
#
# Philosophy:
#   ECG  → tight outlier clip + bandpass only (preserve R-peak sharpness, no smoothing)
#   EMG  → moderate clip + wide bandpass (high-freq muscle content must be kept)
#   EDA  → gentle clip + heavy smooth (slow signal, noise looks like spikes)
#   RESP → moderate clip + narrow bandpass around breathing frequency
#   PPG  → moderate clip + bandpass (pulsatile, similar to ECG but slower)
#   SKT/TEMP → gentle clip + Savitzky-Golay (very slow drift, smooth is correct)
#
SIGNAL_FILTER_PARAMS: dict[str, dict] = {
    "ECG":  {"iqr_factor": 3.0,  "bandpass": (0.5,  40.0), "savgol": None},
    "BVP":  {"iqr_factor": 3.0,  "bandpass": (0.5,  40.0), "savgol": None},
    "EMG":  {"iqr_factor": 4.0,  "bandpass": (20.0, 450.0),"savgol": None},
    "PPG":  {"iqr_factor": 3.5,  "bandpass": (0.5,  10.0), "savgol": None},
    "EDA":  {"iqr_factor": 4.0,  "bandpass": None,          "savgol": (51, 3)},
    "GSR":  {"iqr_factor": 4.0,  "bandpass": None,          "savgol": (51, 3)},
    "RESP": {"iqr_factor": 3.5,  "bandpass": (0.05,  2.0),  "savgol": None},
    "SKT":  {"iqr_factor": 5.0,  "bandpass": None,          "savgol": (101, 2)},
    "TEMP": {"iqr_factor": 5.0,  "bandpass": None,          "savgol": (101, 2)},
    "ACC":  {"iqr_factor": 5.0,  "bandpass": None,          "savgol": None},
    "HR":   {"iqr_factor": 4.0,  "bandpass": None,          "savgol": (11, 2)},
    # Default fallback for any unlisted signal
    "_default": {"iqr_factor": 4.0, "bandpass": None, "savgol": None},
}