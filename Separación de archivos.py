import csv
import os

# Carpeta raíz (la que tú pasas)
carpeta_entrada = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base3_Sujeto5"

# Recorrer todas las subcarpetas y archivos
for root, dirs, files in os.walk(carpeta_entrada):
    for archivo in files:
        if archivo.endswith("_dat.csv"):
            ruta_archivo = os.path.join(root, archivo)
            
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

            # Guardar en la MISMA subcarpeta (root)
            for nombre, datos in zip(encabezados, columnas):
                nombre_archivo = nombre.strip().replace(" ", "_") + ".csv"
                ruta_salida = os.path.join(root, nombre_archivo)
                
                with open(ruta_salida, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Escribir encabezado si el archivo está vacío
                    if os.stat(ruta_salida).st_size == 0:
                        writer.writerow([nombre])
                    
                    for dato in datos:
                        writer.writerow([dato])

print("Procesamiento completo en todas las subcarpetas.")