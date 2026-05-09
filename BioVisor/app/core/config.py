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
