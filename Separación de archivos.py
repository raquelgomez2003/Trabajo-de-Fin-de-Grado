import csv
import os

# Lista de archivos CSV de entrada
archivos_csv = [r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002\sub-cp003_ses-20210206_task-rest_stream-lslhtcviveeye_feat-chunk_level-000_run-002_dat.csv", r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002\sub-cp003_ses-20210206_task-rest_stream-lslshimmerecg_feat-chunk_level-000_run-002_dat.csv", r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002\sub-cp003_ses-20210206_task-rest_stream-lslshimmereda_feat-chunk_level-000_run-002_dat.csv", r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002\sub-cp003_ses-20210206_task-rest_stream-lslshimmeremg_feat-chunk_level-000_run-002_dat.csv", r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002\sub-cp003_ses-20210206_task-rest_stream-lslshimmerresp_feat-chunk_level-000_run-002_dat.csv", r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002\sub-cp003_ses-20210206_task-rest_stream-lslxp11xpcac_feat-chunk_level-000_run-002_dat.csv", r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002\sub-cp003_ses-20210206_task-rest_stream-lslxp11xpcplt_feat-chunk_level-000_run-002_dat.csv"]

# Carpeta donde se guardarán los archivos
carpeta_salida = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-002"
os.makedirs(carpeta_salida, exist_ok=True)

for archivo_csv in archivos_csv:
    with open(archivo_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Primera fila → nombres de columnas
        encabezados = next(reader)
        
        # Crear listas para cada columna
        columnas = [[] for _ in encabezados]
        
        # Leer datos
        for fila in reader:
            for i, valor in enumerate(fila):
                columnas[i].append(valor)

    # Guardar columnas
    for nombre, datos in zip(encabezados, columnas):
        nombre_archivo = nombre.strip().replace(" ", "_") + ".csv"
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
        
        # ⚠️ IMPORTANTE: modo 'a' para no sobrescribir
        with open(ruta_salida, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Si el archivo está vacío, escribir encabezado
            if os.stat(ruta_salida).st_size == 0:
                writer.writerow([nombre])
            
            for dato in datos:
                writer.writerow([dato])

print("Todos los archivos procesados correctamente.")