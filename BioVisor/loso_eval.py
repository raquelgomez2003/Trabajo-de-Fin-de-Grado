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
BASE_FOLDER = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1"      # r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1"
BASE_NAME   = "Base1"                  # nombre base -> {BASE_NAME}_Sujeto{n} Base1
DEVICE      = "Biopac" #Biopac o Empatica
N_SUBJECTS  = 5 #5 o 10
#SIGNALS     = ["BVP", "EDA", "TEMP", "ACC", "HR"]      #setup Empatica
#FS_MAP      = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32, "HR": 1}

SIGNALS     = ["ECG", "EDA", "EMG", "PPG", "RESP", "SKT"]   # setup Biopac
FS_MAP      = {"ECG": 2000, "EDA": 2000, "EMG": 2000, "PPG": 2000, "RESP": 2000, "SKT": 2000}
WINDOW_SEC  = 60.0
STEP_SEC    = 30.0
APPLY_FILTER = True

# Intervalos de estres (vexing) por sujeto, en segundos: {n_sujeto: (inicio, fin)}
# Rellena con los tiempos reales de cada sujeto de la Base 2.
VEXING = {
    1:  (1613, 2579),
    2:  (2268, 3233),
    3:  (1519, 2461),
    4:  (1689, 2642),
    5:  (1793, 2761),
}
# Intervalos de calming por sujeto (para acotar la ventana del protocolo).
CALMING = {
    1:  (502, 1469),
    2:  (1166, 2127),
    3:  (425, 1374),
    4:  (582, 1505),
    5:  (669, 1638),
}
# ─────────────────────────────────────────────────────────────────────────────

# Intervalos de estres (vexing) por sujeto, en segundos: {n_sujeto: (inicio, fin)}
#VEXING = {
#    1:  (1613, 2579),
#    2:  (2268, 3233),
#    3:  (1519, 2461),
#    4:  (1689, 2642),
#    5:  (1793, 2761),
#    1:  (2622, 3515),
#    2:  (2895, 3824),
#    3:  (2688, 3613),
#    4:  (3065, 4002),
#    5:  (2714, 3649),
#    6:  (2908, 3854),
#    7:  (3058, 3954),
#    8:  (2714, 3629),
#    9:  (2689, 3604),
#    10: (2709, 3655),
#}
#CALMING = {
#    1:  (1336, 2234),
#    2:  (1582, 2506),
#    3:  (1372, 2299),
#    4:  (1738, 2664),
#    5:  (1376, 2319),
#    6:  (1531, 2476),
#    7:  (1744, 2646),
#    8:  (1382, 2318),
#    9:  (1379, 2296),
#    10: (1373, 2319),
#}
# Intervalos de calming por sujeto (para acotar la ventana del protocolo).
#VEXING = {
#    1:  (1613, 2579),
#    2:  (2268, 3233),
#    3:  (1519, 2461),
#    4:  (1689, 2642),
#    5:  (1793, 2761),
#}
#CALMING = {
#    1:  (502, 1469),
#    2:  (1166, 2127),
#    3:  (425, 1374),
#    4:  (582, 1505),
#    5:  (669, 1638),
#}

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

def _zscore_per_subject(X: np.ndarray) -> np.ndarray:
    """Z-score POR COLUMNA usando la media/desv. del PROPIO sujeto.
    Elimina la linea base individual (clave para la generalizacion LOSO)."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)      # columnas constantes -> no dividir por 0
    return (X - mu) / sd

def evaluate_signal(sig_name: str) -> float | None:
    """LOSO-AUC por señal. Normaliza (z-score) por sujeto y por columna,
    y promedia el AUC de cada sujeto (macro), NO un unico AUC agrupado."""
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

        y    = build_labels_from_intervals(t, vex).astype(int)
        mask = (t >= t_lo) & (t <= t_hi)

        Xs = _zscore_per_subject(X[mask])       # <-- z-score POR SUJETO y POR COLUMNA
        X_list.append(Xs)
        y_list.append(y[mask])
        g_list.append(np.full(mask.sum(), num))  # grupo = sujeto

    if not X_list:
        return None

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    g = np.concatenate(g_list)

    if len(np.unique(y)) < 2 or len(np.unique(g)) < 2:
        return None

    logo  = LeaveOneGroupOut()
    proba = np.zeros(len(y), dtype=float)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in logo.split(X, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                continue
            pipe = _make_pipeline()
            pipe.fit(X[tr], y[tr])
            proba[te] = pipe.predict_proba(X[te])[:, 1]

    # ── AUC POR SUJETO y luego media (macro), NO un unico AUC agrupado ─────────
    aucs, detail = [], []
    for subj in np.unique(g):
        m = g == subj
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue                    # sujeto sin las dos clases -> AUC indefinido
        a = roc_auc_score(y[m], proba[m])
        aucs.append(a)
        detail.append(f"S{int(subj)}={a:.2f}")

    if not aucs:
        return None

    print(f"    [{sig_name}] por sujeto: " + "  ".join(detail))
    print(f"    [{sig_name}] media={np.mean(aucs):.3f}  sd={np.std(aucs):.3f}  n={len(aucs)}")
    return float(np.mean(aucs))

def evaluate_all() -> float | None:
    """LOSO con TODAS las señales FUSIONADAS: un vector de características por
    ventana que concatena las features de cada señal. Da UN AUC por sujeto
    (todas las señales juntas) y promedia. Esto es lo que pide '1 por sujeto'."""
    # 1) Cargar y extraer features de cada sujeto/señal
    raw = {}   # num -> {sig: (X, t)}
    for num in range(1, N_SUBJECTS + 1):
        signals = _load_subject(num)
        d = {}
        for sig in SIGNALS:
            if sig not in signals:
                continue
            fs = FS_MAP.get(sig, 2000)
            X, t = extract_features_windowed(signals[sig], fs, sig,
                                             WINDOW_SEC, STEP_SEC)
            if len(X):
                d[sig] = (np.asarray(X, float), np.asarray(t, float))
        if d:
            raw[num] = d

    if not raw:
        return None

    # 2) Ancho (nº de columnas) de cada señal -> fijo para todos los sujetos
    width = {}
    for d in raw.values():
        for sig, (X, _) in d.items():
            width[sig] = X.shape[1]
    ordered    = [s for s in SIGNALS if s in width]   # orden fijo de columnas

    # 3) Matriz combinada por sujeto (señal ausente -> NaN, ya la imputa el pipeline)
    X_list, y_list, g_list = [], [], []
    for num, d in raw.items():
        # nº de ventanas comun = minimo entre las señales presentes del sujeto.
        # t_centers[i] es IDENTICO entre señales (misma ventana/paso en segundos,
        # independiente de fs), asi que la ventana i de BVP == ventana i de EDA.
        n_win = min(X.shape[0] for X, _ in d.values())
        t     = next(iter(d.values()))[1][:n_win]

        cols = []
        for sig in ordered:
            if sig in d:
                cols.append(d[sig][0][:n_win, :])
            else:
                cols.append(np.full((n_win, width[sig]), np.nan))
        Xc = np.hstack(cols)   # (n_win, total de columnas)

        vex  = [VEXING[num]]  if num in VEXING  and VEXING[num][1]  > VEXING[num][0]  else []
        calm = [CALMING[num]] if num in CALMING and CALMING[num][1] > CALMING[num][0] else []
        prot = vex + calm
        if not prot:
            continue
        t_lo = min(s for s, _ in prot)
        t_hi = max(e for _, e in prot)

        y    = build_labels_from_intervals(t, vex).astype(int)
        mask = (t >= t_lo) & (t <= t_hi)
        if mask.sum() == 0:
            continue

        Xc = _zscore_per_subject(Xc[mask])   # z-score por sujeto y por columna
        X_list.append(Xc)
        y_list.append(y[mask])
        g_list.append(np.full(mask.sum(), num))

    if not X_list:
        return None

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    g = np.concatenate(g_list)

    if len(np.unique(y)) < 2 or len(np.unique(g)) < 2:
        return None

    logo  = LeaveOneGroupOut()
    proba = np.zeros(len(y), dtype=float)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in logo.split(X, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                continue
            pipe = _make_pipeline()
            pipe.fit(X[tr], y[tr])
            proba[te] = pipe.predict_proba(X[te])[:, 1]

    # ── UN AUC por sujeto (todas las señales) y luego media (macro) ────────────
    aucs, detail = [], []
    for subj in np.unique(g):
        m = g == subj
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], proba[m])
        aucs.append(a)
        detail.append(f"S{int(subj)}={a:.2f}")

    if not aucs:
        return None

    print(f"  Señales fusionadas: {ordered}")
    print("  AUC por sujeto:  " + "  ".join(detail))
    print(f"  media={np.mean(aucs):.3f}  sd={np.std(aucs):.3f}  n={len(aucs)}")
    return float(np.mean(aucs))

def main():
    print(f"LOSO evaluation — {DEVICE} — {N_SUBJECTS} sujetos — TODAS las señales\n")
    auc = evaluate_all()
    if auc is None:
        print("Sin datos suficientes.")
        return
    print(f"\nAUC LOSO (todas las señales, media por sujeto) = {auc:.3f}")

    out = "loso_auc_base2.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "auc_loso"])
        w.writerow(["todas_las_senales_fusionadas", f"{auc:.4f}"])
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()