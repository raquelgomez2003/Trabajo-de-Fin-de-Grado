import csv
import os

# Ruta del archivo CSV de entrada
archivo_csv = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-001\sub-cp003_ses-20210206_task-rest_stream-lslshimmereda_feat-chunk_level-000_run-001_dat.csv"

# Carpeta donde se guardarán los archivos (se crea si no existe)
carpeta_salida = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-000_run-001"
os.makedirs(carpeta_salida, exist_ok=True)

# Leer el CSV
with open(archivo_csv, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    
    # Primera fila → nombres de columnas
    encabezados = next(reader)
    
    # Crear listas para cada columna
    columnas = [[] for _ in encabezados]
    
    # Leer datos y separarlos por columna
    for fila in reader:
        for i, valor in enumerate(fila):
            columnas[i].append(valor)

# Guardar cada columna en un archivo independiente
for nombre, datos in zip(encabezados, columnas):
    # Limpiar nombre (por si hay espacios o caracteres raros)
    nombre_archivo = nombre.strip().replace(" ", "_") + ".csv"
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    
    with open(ruta_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escribir encabezado
        writer.writerow([nombre])
        
        # Escribir datos
        for dato in datos:
            writer.writerow([dato])

print("Archivos generados correctamente.")