import os

# Ruta del archivo .py que quieres comprobar
ruta_archivo = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\EDA\Subject3F_EDA.py"

# Comprobar si existe
if os.path.exists(ruta_archivo):
    print("El archivo existe")
    
    # Mostrar tamaño
    tamaño = os.path.getsize(ruta_archivo)
    print(f"Tamaño del archivo: {tamaño} bytes")
    
    # Mostrar primeras líneas
    print("\nPrimeras líneas del archivo:\n")
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        for i in range(10):  # Muestra las primeras 10 líneas
            linea = f.readline()
            if not linea:
                break
            print(linea.strip())
else:
    print("El archivo NO existe")