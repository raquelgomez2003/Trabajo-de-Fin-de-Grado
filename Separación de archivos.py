import csv
import os

# Carpeta donde están los archivos
carpeta_entrada = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto1\level-04B_run-009"

# Carpeta de salida
carpeta_salida = carpeta_entrada
os.makedirs(carpeta_salida, exist_ok=True)

# Recorrer todos los archivos
for archivo in os.listdir(carpeta_entrada):
    if archivo.endswith("_dat.csv"):
        ruta_archivo = os.path.join(carpeta_entrada, archivo)
        
        with open(ruta_archivo, newline='', encoding='utf-8') as f:
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
            
            with open(ruta_salida, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Escribir encabezado si el archivo está vacío
                if os.stat(ruta_salida).st_size == 0:
                    writer.writerow([nombre])
                
                for dato in datos:
                    writer.writerow([dato])

print("Todos los archivos *_dat.csv han sido procesados.")