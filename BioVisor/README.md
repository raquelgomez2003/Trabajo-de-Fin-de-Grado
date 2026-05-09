# BioVisor — Biomedical Signal Viewer & Analyser

TFG project: multimodal physiological signal visualisation and stress classification.
Supports **Biopac** and **Empatica E4** datasets.

---

## Features

| Module | What it does |
|--------|-------------|
| **Signal Viewer** | Displays all signals simultaneously with phase shading (baseline / calming / relax / stress) and predicted-stress markers |
| **ECG BPM overlay** | Computes beats per minute via R-peak detection and overlays on ECG panel |
| **Analysis window** | Trains a Random Forest per signal; shows stress markers, feature boxplots, and ROC curves |
| **Setup wizard** | Folder picker + device selector + per-signal Fs config — no hardcoded paths |

---

## Project Structure

```
BioVisor/
│
├── main.py                        ← Entry point: python main.py
│
├── app/                           ✅ ACTIVE — production code
│   ├── core/
│   │   ├── config.py              Device presets, phase colours, RF params
│   │   ├── data_loader.py         CSV loading for Biopac & Empatica
│   │   ├── models.py              Feature extraction, StressModel, BPM
│   │   └── plotter.py             Signal plot, boxplots, ROC figures
│   └── gui/
│       ├── app_window.py          Main window (orchestrator)
│       ├── setup_window.py        Setup wizard (folder, device, Fs)
│       ├── viewer_window.py       Signal Viewer tab
│       └── analysis_window.py     Analysis tab (boxplots + ROC)
│
├── tests/                         ✅ KEEP — run with pytest
│   └── test_models.py
│
├── archive/                       ⚠️  PROTOTYPES — do not use
│   └── *_v1.py                    Early single-file scripts
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python main.py
```

On first launch the **Setup Wizard** appears:
1. Select your dataset root folder
2. Choose device (Biopac / Empatica)
3. Set number of subjects and sampling frequencies
4. Click **Confirm & Open Viewer**

Then:
- Click a **Subject** button to load signals
- Enter **Calming** and **Stress** interval times (seconds)
- Click **Update Viewer** to see signals with phase shading
- Click **▶ Run Analysis** to train the classifier and view boxplots + ROC curves

---

## Supported Signals

| Device | Signals | Default Fs |
|--------|---------|-----------|
| Biopac | ECG, EDA, EMG, PPG, RESP, SKT | 2000 Hz each |
| Empatica E4 | BVP, EDA, TEMP, ACC, HR | 64 / 4 / 4 / 32 / 1 Hz |

> Frequencies are editable in the Setup Wizard.

---

## Analysis Details

**Features per window** (default: 60 s / 50 % overlap):
Mean · Std · RMS · Peak-to-peak · Skewness · Kurtosis · Low-freq power · High-freq power

**Classifier**: `RandomForestClassifier` (200 trees, balanced class weights)

**Output per signal**:
- Stress markers overlaid on the signal (red ▼ triangles)
- Boxplots: feature distributions for Baseline / Calming / Stress phases
- ROC curve with AUC and optimal Youden's-J threshold

**ECG BPM**: R-peak detection via bandpass filter (0.5–40 Hz) + `scipy.signal.find_peaks`, plotted as a dashed orange line on the ECG panel.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dependencies

```
customtkinter  matplotlib  numpy  pandas  scikit-learn  scipy  joblib
```
