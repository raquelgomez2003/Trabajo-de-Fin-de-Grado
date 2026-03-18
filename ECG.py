import os

# Carpeta raíz (la que contiene todas las subcarpetas)
carpeta_raiz = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data"

# Recorre todas las carpetas y subcarpetas
for ruta_actual, subcarpetas, archivos in os.walk(carpeta_raiz):
    
    for archivo in archivos:
        
        if not archivo.endswith(".py"):  # Evita convertir los .py
            ruta_original = os.path.join(ruta_actual, archivo)
            
            nombre_base = os.path.splitext(archivo)[0]
            nuevo_nombre = nombre_base + ".py"
            ruta_nueva = os.path.join(ruta_actual, nuevo_nombre)
            
            try:
                with open(ruta_original, "r", encoding="utf-8") as f:
                    contenido = f.read()
                
                with open(ruta_nueva, "w", encoding="utf-8") as f:
                    f.write(contenido)
                
                print(f"Convertido: {ruta_original} → {ruta_nueva}")
            
            except Exception as e:
                print(f"No se pudo convertir {ruta_original}. Error: {e}")