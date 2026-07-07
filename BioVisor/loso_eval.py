"""
loso_eval_base2.py
Evaluacion Leave-One-Subject-Out (LOSO) para la Base de datos 2 (Empatica).

Entrena con N-1 sujetos y evalua en el sujeto excluido, rotando sobre todos.
Reutiliza la extraccion de caracteristicas y el etiquetado por protocolo de
models.py. NO modifica la aplicacion.

Uso:
    python loso_eval_base2.py
"""

from __future__ import annotations
import os
import csv
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from app.core.data_loader import load_subject_folder
from app.core.models import extract_features_windowed, build_labels_from_intervals

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION — AJUSTA ESTO A TU CASO
# ─────────────────────────────────────────────────────────────────────────────
BASE_FOLDER = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base2"      # carpeta que seleccionas en el setup
BASE_NAME   = "Base2"                  # nombre base -> {BASE_NAME}_Sujeto{n}
DEVICE      = "Empatica"
N_SUBJECTS  = 10
SIGNALS     = ["BVP", "EDA", "TEMP", "ACC", "HR"]      # las de tu setup Empatica
FS_MAP      = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32, "HR": 1}
WINDOW_SEC  = 60.0
STEP_SEC    = 30.0
APPLY_FILTER = True

# Intervalos de estres (vexing) por sujeto, en segundos: {n_sujeto: (inicio, fin)}
# Rellena con los tiempos reales de cada sujeto de la Base 2.
VEXING = {
    1:  (2622, 3515),
    2:  (2895, 3824),
    3:  (2688, 3613),
    4:  (3065, 4002),
    5:  (2714, 3649),
    6:  (2908, 3854),
    7:  (3058, 3954),
    8:  (2714, 3629),
    9:  (2689, 3604),
    10: (2709, 3655),
}
# Intervalos de calming por sujeto (para acotar la ventana del protocolo).
CALMING = {
    1:  (1336, 2234),
    2:  (1582, 2506),
    3:  (1372, 2299),
    4:  (1738, 2664),
    5:  (1376, 2319),
    6:  (1531, 2476),
    7:  (1744, 2646),
    8:  (1382, 2318),
    9:  (1379, 2296),
    10: (1373, 2319),
}
# ─────────────────────────────────────────────────────────────────────────────


def _subject_dir(num: int) -> str:
    return os.path.join(BASE_FOLDER, f"{BASE_NAME}_Sujeto{num}")


def _candidates(num: int) -> list[str]:
    d = _subject_dir(num)
    if DEVICE == "Biopac":
        return [os.path.join(d, "Biopac data"), d]
    return [os.path.join(d, "Empatica_data"),
            os.path.join(d, "Empatica data"),
            os.path.join(d, "Empatica"), d]


def _load_subject(num: int) -> dict:
    for cand in _candidates(num):
        if os.path.isdir(cand):
            try:
                loaded = load_subject_folder(
                    cand, DEVICE, SIGNALS, fs_map=dict(FS_MAP),
                    apply_filters=APPLY_FILTER)
                if loaded:
                    return loaded
            except Exception as e:
                print(f"[SKIP] {cand}: {e}")
    return {}


def _make_pipeline():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])


def evaluate_signal(sig_name: str) -> float | None:
    """LOSO-AUC para una señal, agregando las ventanas de todos los sujetos."""
    X_list, y_list, g_list = [], [], []

    for num in range(1, N_SUBJECTS + 1):
        signals = _load_subject(num)
        if sig_name not in signals:
            continue
        fs = FS_MAP.get(sig_name, 2000)
        X, t = extract_features_windowed(signals[sig_name], fs, sig_name,
                                         WINDOW_SEC, STEP_SEC)
        if len(X) == 0:
            continue

        vex  = [VEXING[num]]  if num in VEXING  and VEXING[num][1]  > VEXING[num][0]  else []
        calm = [CALMING[num]] if num in CALMING and CALMING[num][1] > CALMING[num][0] else []
        prot = vex + calm
        if not prot:
            continue
        t_lo = min(s for s, _ in prot)
        t_hi = max(e for _, e in prot)

        y    = build_labels_from_intervals(t, vex).astype(int)  # vexing=1, resto=0
        mask = (t >= t_lo) & (t <= t_hi)                        # solo protocolo

        X_list.append(X[mask])
        y_list.append(y[mask])
        g_list.append(np.full(mask.sum(), num))                 # grupo = sujeto

    if not X_list:
        return None

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    g = np.concatenate(g_list)

    if len(np.unique(y)) < 2 or len(np.unique(g)) < 2:
        return None

    # LOSO: cada fold deja fuera un sujeto entero
    logo   = LeaveOneGroupOut()
    proba  = np.zeros(len(y), dtype=float)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in logo.split(X, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                continue
            pipe = _make_pipeline()
            pipe.fit(X[tr], y[tr])
            proba[te] = pipe.predict_proba(X[te])[:, 1]

    return float(roc_auc_score(y, proba))


def main():
    print(f"LOSO evaluation — {DEVICE} — {N_SUBJECTS} sujetos\n")
    results = {}
    for sig in SIGNALS:
        auc = evaluate_signal(sig)
        results[sig] = auc
        if auc is None:
            print(f"  {sig:5s} : sin datos suficientes")
        else:
            print(f"  {sig:5s} : AUC LOSO = {auc:.3f}")

    # Guardar CSV
    out = "loso_auc_base2.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal", "auc_loso"])
        for sig, auc in results.items():
            w.writerow([sig, "" if auc is None else f"{auc:.4f}"])
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()