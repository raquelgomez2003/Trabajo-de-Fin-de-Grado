import numpy as np
import os

def detectar_tipo(nombre):
    nombre = nombre.upper()
    if "ECG" in nombre: return "ECG"
    if "EDA" in nombre: return "EDA"
    if "EMG" in nombre: return "EMG"
    if "PPG" in nombre: return "PPG"
    if "RESP" in nombre: return "RESP"
    if "SKT" in nombre: return "SKT"
    return None

def cargar_sujeto(base_path, num):
    ruta = os.path.join(base_path, f"Base1_Sujeto{num}", "Biopac data")

    if not os.path.exists(ruta):
        raise Exception(f"No existe: {ruta}")

    senales = {}

    for archivo in os.listdir(ruta):
        if archivo.endswith(".csv"):
            tipo = detectar_tipo(archivo)
            if tipo:
                try:
                    data = np.loadtxt(os.path.join(ruta, archivo), delimiter=",", skiprows=1)
                    senales[tipo] = data
                except:
                    pass

    return senales