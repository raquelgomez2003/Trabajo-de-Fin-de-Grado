"""
tests/test_models.py
Unit tests for feature extraction, label building, and BPM.
Run with: pytest tests/
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.models import (
    extract_features_windowed,
    build_labels,
    StressModel,
    compute_bpm,
    FEATURE_NAMES,
)


# ── Feature extraction ─────────────────────────────────────────────────────────

def test_extract_features_shape():
    fs     = 2000
    signal = np.random.randn(int(fs * 180))   # 3 minutes
    X, t   = extract_features_windowed(signal, fs, window_sec=60, step_sec=30)
    # 3 min with 60-s window / 30-s step → 3 windows
    assert X.shape[0] == 3
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(t) == 3


def test_extract_features_too_short():
    fs     = 2000
    signal = np.random.randn(100)
    X, t   = extract_features_windowed(signal, fs, window_sec=60, step_sec=30)
    assert X.shape[0] == 0


def test_feature_names_length():
    assert len(FEATURE_NAMES) == 8


# ── Label builder ─────────────────────────────────────────────────────────────

def test_build_labels_basic():
    t = np.array([10, 30, 50, 70, 90], dtype=float)
    labels = build_labels(t, [(25, 75)])
    expected = np.array([0, 1, 1, 1, 0])
    np.testing.assert_array_equal(labels, expected)


def test_build_labels_no_stress():
    t      = np.arange(10, dtype=float)
    labels = build_labels(t, [])
    assert labels.sum() == 0


# ── Stress model ──────────────────────────────────────────────────────────────

def test_stress_model_trains_and_predicts():
    rng = np.random.default_rng(0)
    X   = rng.random((40, 8))
    y   = np.array([0] * 20 + [1] * 20)
    model = StressModel(n_estimators=20, random_state=0)
    model.fit(X, y)
    preds  = model.predict(X)
    scores = model.predict_proba(X)
    assert preds.shape == (40,)
    assert scores.shape == (40,)
    assert set(preds).issubset({0, 1})
    assert scores.min() >= 0 and scores.max() <= 1


# ── BPM ───────────────────────────────────────────────────────────────────────

def test_compute_bpm_synthetic_ecg():
    """Synthetic ECG: sharp peaks every 1 s → ~60 BPM."""
    fs   = 2000
    t    = np.arange(fs * 120)          # 2 minutes
    ecg  = np.zeros(len(t))
    peak_interval = fs                  # 1 peak/second
    ecg[::peak_interval] = 1.0

    bpm_t, bpm_v = compute_bpm(ecg, fs, window_sec=10, step_sec=5)
    assert len(bpm_t) > 0
    np.testing.assert_allclose(bpm_v, 60.0, atol=5)


def test_compute_bpm_too_short():
    fs  = 2000
    ecg = np.zeros(100)
    t, v = compute_bpm(ecg, fs)
    assert len(t) == 0 and len(v) == 0
