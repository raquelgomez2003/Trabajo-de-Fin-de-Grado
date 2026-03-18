import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

# Ruta del archivo (cámbiala por la tuya)
señal = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\ECG\Subject3F_ECG.py"

# Cargar la señal
x = np.loadtxt(señal)

# Frecuencia de muestreo original
fs = 2000

# Diezmado
y = decimate(x,4)

# Nueva frecuencia de muestreo
fs_nueva = fs /2 #pasa a ser 500Hz

# Eje temporal  
t = np.arange(len(y)) / fs_nueva

# Representación
plt.plot(t, y)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
plt.title("ECG diezmado")
plt.show()