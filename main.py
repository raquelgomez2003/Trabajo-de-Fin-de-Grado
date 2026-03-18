import os
import glob
import pandas as pd
import numpy as np

carpeta_csv = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\ECG"
carpeta_npy = r"C:\Users\raque\Desktop\TFG\bse de datos TFG\Biopac data\ECG\python"

# Crear carpeta si no existe
os.makedirs(carpeta_npy, exist_ok=True)

# Obtener todos los .csv
archivos = glob.glob(os.path.join(carpeta_csv, "*.csv"))

columnas = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

for ruta_csv in archivos:
    try:
        # Leer CSV
        datos = pd.read_csv(
            ruta_csv,
            header=0,
            names=columnas,
            delimiter=','
        )

        # Convertir a array NumPy
        datos_np = datos.values

        # Guardar en formato .npy
        nombre_base = os.path.basename(ruta_csv).replace(".csv", ".npy")
        ruta_npy = os.path.join(carpeta_npy, nombre_base)

        np.save(ruta_npy, datos_np)

        print("Guardado:", nombre_base)

    except Exception as e:
        print("Error al procesar", ruta_csv, ":", e) 
