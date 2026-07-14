"""
loso_eval.py
Evaluacion Leave-One-Subject-Out (LOSO) para las dos bases de datos.

Selecciona la base con la variable BASE:
    BASE = 1  ->  Biopac    (5 sujetos)
    BASE = 2  ->  Empatica  (10 sujetos)

Entrena con N-1 sujetos y evalua en el sujeto excluido, rotando sobre todos.
Reutiliza la extraccion de caracteristicas y el etiquetado por protocolo de
models.py. NO modifica la aplicacion.

Uso:
    # cambia BASE = 1 o BASE = 2 abajo y ejecuta una vez por cada base
    python loso_eval.py
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
# Al ejecutar se evaluan LAS DOS bases seguidas (Biopac y Empatica).
# No hay que elegir nada: salen los dos CSV en una sola ejecucion.
# ─────────────────────────────────────────────────────────────────────────────

# Parametros comunes a las dos bases
WINDOW_SEC   = 60.0
STEP_SEC     = 30.0
APPLY_FILTER = True

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION POR BASE
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    # ── Base 1 — Biopac — 5 sujetos ──────────────────────────────────────────
    1: {
        "BASE_FOLDER": r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1",
        "BASE_NAME":   "Base1",
        "DEVICE":      "Biopac",
        "N_SUBJECTS":  5,
        "SIGNALS":     ["ECG", "EDA", "EMG", "PPG", "RESP", "SKT"],
        "FS_MAP":      {"ECG": 2000, "EDA": 2000, "EMG": 2000,
                        "PPG": 2000, "RESP": 2000, "SKT": 2000},
        "VEXING": {
            1: (1613, 2579),
            2: (2268, 3233),
            3: (1519, 2461),
            4: (1689, 2642),
            5: (1793, 2761),
        },
        "CALMING": {
            1: (502, 1469),
            2: (1166, 2127),
            3: (425, 1374),
            4: (582, 1505),
            5: (669, 1638),
        },
    },

    # ── Base 2 — Empatica — 10 sujetos ───────────────────────────────────────
    2: {
        "BASE_FOLDER": r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base2",
        "BASE_NAME":   "Base2",
        "DEVICE":      "Empatica",
        "N_SUBJECTS":  10,
        "SIGNALS":     ["BVP", "EDA", "TEMP", "ACC", "HR"],
        "FS_MAP":      {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32, "HR": 1},
        "VEXING": {
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
        },
        "CALMING": {
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
        },
    },
}

# ── Nombres globales que usan las funciones de abajo (se rellenan por base) ───
BASE        = None
BASE_FOLDER = None
BASE_NAME   = None
DEVICE      = None
N_SUBJECTS  = None
SIGNALS     = None
FS_MAP      = None
VEXING      = None
CALMING     = None


def _apply_config(base: int) -> None:
    """Vuelca la config de la base indicada a las variables globales que usan
    las funciones de evaluacion. Permite recorrer las dos bases en una ejecucion."""
    global BASE, BASE_FOLDER, BASE_NAME, DEVICE, N_SUBJECTS
    global SIGNALS, FS_MAP, VEXING, CALMING
    _SUBJECT_CACHE.clear()          # datos de la base anterior fuera
    cfg         = CONFIGS[base]
    BASE        = base
    BASE_FOLDER = cfg["BASE_FOLDER"]
    BASE_NAME   = cfg["BASE_NAME"]
    DEVICE      = cfg["DEVICE"]
    N_SUBJECTS  = cfg["N_SUBJECTS"]
    SIGNALS     = cfg["SIGNALS"]
    FS_MAP      = cfg["FS_MAP"]
    VEXING      = cfg["VEXING"]
    CALMING     = cfg["CALMING"]
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


# Cache de sujetos ya cargados (para NO releer/refiltrar los CSV en cada señal).
# Se vacia en _apply_config al cambiar de base.
_SUBJECT_CACHE: dict[int, dict] = {}


def _load_subject(num: int) -> dict:
    if num in _SUBJECT_CACHE:                 # ya cargado en esta base -> reutiliza
        return _SUBJECT_CACHE[num]
    for cand in _candidates(num):
        if os.path.isdir(cand):
            try:
                loaded = load_subject_folder(
                    cand, DEVICE, SIGNALS, fs_map=dict(FS_MAP),
                    apply_filters=APPLY_FILTER)
                if loaded:
                    _SUBJECT_CACHE[num] = loaded
                    return loaded
            except Exception as e:
                print(f"[SKIP] {cand}: {e}")
    _SUBJECT_CACHE[num] = {}                   # cachea el fallo -> no reintenta cada señal
    return {}


def _make_pipeline():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _zscore_per_subject(X: np.ndarray) -> np.ndarray:
    """Z-score POR COLUMNA usando la media/desv. del PROPIO sujeto.
    Elimina la linea base individual (clave para la generalizacion LOSO)."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)      # columnas constantes -> no dividir por 0
    return (X - mu) / sd


def evaluate_signal(sig_name: str) -> dict | None:
    """LOSO-AUC por señal. Normaliza (z-score) por sujeto y por columna.
    Devuelve {n_sujeto: auc} (o None si no hay datos)."""
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
    per_subj, aucs, detail = {}, [], []
    for subj in np.unique(g):
        m = g == subj
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue                    # sujeto sin las dos clases -> AUC indefinido
        a = float(roc_auc_score(y[m], proba[m]))
        per_subj[int(subj)] = a
        aucs.append(a)
        detail.append(f"S{int(subj)}={a:.2f}")

    if not aucs:
        return None

    print(f"    [{sig_name}] por sujeto: " + "  ".join(detail))
    print(f"    [{sig_name}] media={np.mean(aucs):.3f}  sd={np.std(aucs):.3f}  n={len(aucs)}")
    return per_subj


def evaluate_all() -> dict | None:
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
    per_subj, aucs, detail = {}, [], []
    for subj in np.unique(g):
        m = g == subj
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        a = float(roc_auc_score(y[m], proba[m]))
        per_subj[int(subj)] = a
        aucs.append(a)
        detail.append(f"S{int(subj)}={a:.2f}")

    if not aucs:
        return None

    print(f"  Señales fusionadas: {ordered}")
    print("  AUC por sujeto:  " + "  ".join(detail))
    print(f"  media={np.mean(aucs):.3f}  sd={np.std(aucs):.3f}  n={len(aucs)}")
    return per_subj


def run_base(base: int) -> None:
    """Evalua una base completa (AUC por señal + fusionadas) y guarda su CSV."""
    _apply_config(base)
    print(f"\n{'='*70}")
    print(f"LOSO evaluation — BASE {BASE} — {DEVICE} — {N_SUBJECTS} sujetos")
    print(f"Carpeta: {BASE_FOLDER}")
    print(f"{'='*70}\n")

    # 1) AUC POR SEÑAL (tablas 6.5 / 6.6)
    print("── AUC LOSO por señal ─────────────────────────────────────────")
    per_signal = {}
    for sig in SIGNALS:
        auc_sig = evaluate_signal(sig)
        if auc_sig is not None:
            per_signal[sig] = auc_sig

    # 2) AUC con TODAS las señales fusionadas
    print("\n── AUC LOSO (todas las señales fusionadas) ────────────────────")
    auc_all = evaluate_all()

    # 3) Guardado dependiente de BASE_NAME (Base1 y Base2 no se pisan).
    #    Dos secciones:
    #      (A) AUC por sujeto -> modelo con TODAS las señales fusionadas
    #      (B) AUC por señal  -> media +/- sd sobre todos los sujetos
    out = f"loso_auc_{BASE_NAME.lower()}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # ── (A) AUC por sujeto (todas las señales fusionadas) ──────────────────
        w.writerow(["# AUC por sujeto (todas las senales fusionadas)"])
        w.writerow(["sujeto", "auc"])
        if auc_all is not None:
            for n in range(1, N_SUBJECTS + 1):
                v = auc_all.get(n)
                w.writerow([f"S{n}", f"{v:.4f}" if v is not None else ""])
            present = [v for v in auc_all.values()]
            w.writerow(["media", f"{np.mean(present):.4f}" if present else ""])
            w.writerow(["sd",    f"{np.std(present):.4f}"  if present else ""])
        w.writerow([])

        # ── (B) AUC por señal (media +/- sd sobre todos los sujetos) ───────────
        w.writerow(["# AUC por senal (media +/- sd sobre todos los sujetos)"])
        w.writerow(["senal", "auc_media", "auc_sd"])
        for sig, per_subj in per_signal.items():
            vals = [v for v in per_subj.values()]
            w.writerow([sig,
                        f"{np.mean(vals):.4f}" if vals else "",
                        f"{np.std(vals):.4f}"  if vals else ""])

    print(f"\nGuardado en {out}")


def main():
    # Recorre TODAS las bases definidas en CONFIGS (1 = Biopac, 2 = Empatica).
    for base in sorted(CONFIGS):
        run_base(base)
    print("\nListo: generados los CSV de todas las bases.")


if __name__ == "__main__":
    main()