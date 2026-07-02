"""
align_hr_ecg.py
Finds the second in the HR (Empatica, 1 Hz) signal where it starts to
coincide with the BPM derived from ECG (Biopac, 2000 Hz).

Method: cross-correlation between ECG-derived BPM (resampled to 1 Hz)
and the Empatica HR signal. The lag at the maximum of the cross-correlation
gives the offset in seconds.

Prints a single integer to stdout: the offset in seconds.

Usage:
    python align_hr_ecg.py <ecg_csv_path> <hr_csv_path>
"""

import sys
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

ECG_PATH = sys.argv[1] if len(sys.argv) > 1 else ""
HR_PATH  = sys.argv[2] if len(sys.argv) > 2 else ""
ECG_FS   = 2000.0


# ── 1. Load ECG and compute BPM at 1 Hz ──────────────────────────────────────
ecg = np.loadtxt(ECG_PATH)

nyq  = ECG_FS / 2.0
b, a = butter(2, [0.5 / nyq, min(40.0 / nyq, 0.99)], btype="band")
ecg_f = filtfilt(b, a, ecg)
ecg_n = ecg_f / np.max(np.abs(ecg_f))

peaks, _ = find_peaks(ecg_n, height=0.3, distance=int(0.4 * ECG_FS))

rr_ms   = (np.diff(peaks) / ECG_FS) * 1000.0
bpm_ecg = 60000.0 / rr_ms
t_peaks = (peaks[:-1] + peaks[1:]) / 2.0 / ECG_FS

valid   = (bpm_ecg >= 30) & (bpm_ecg <= 220)
bpm_ecg = bpm_ecg[valid]
t_peaks = t_peaks[valid]

ecg_dur     = len(ecg) / ECG_FS
t_uniform   = np.arange(0, ecg_dur, 1.0)
bpm_ecg_1hz = np.interp(t_uniform, t_peaks, bpm_ecg)


# ── 2. Load HR (Empatica, 1 Hz) ───────────────────────────────────────────────
with open(HR_PATH) as f:
    lines = f.readlines()
hr_data = np.array([float(l.strip()) for l in lines[2:] if l.strip()])


# ── 3. Cross-correlation ──────────────────────────────────────────────────────
# Zero-mean both signals — cross-correlation works on zero-mean signals
ecg_zm = bpm_ecg_1hz - bpm_ecg_1hz.mean()
hr_zm  = hr_data     - hr_data.mean()

# Full cross-correlation: xcorr[k] = sum(hr[t+k] * ecg[t])
# Positive lag k means HR starts k seconds after ECG start
xcorr = np.correlate(hr_zm, ecg_zm, mode="full")

# Build lag array
n_ecg = len(ecg_zm)
n_hr  = len(hr_zm)
lags  = np.arange(-(n_ecg - 1), n_hr)

# Only consider lags where windows fully overlap (0 to len(hr)-len(ecg))
max_lag = max(0, n_hr - n_ecg)
mask    = (lags >= 0) & (lags <= max_lag)

best_lag = int(lags[mask][np.argmax(xcorr[mask])])

print(best_lag)