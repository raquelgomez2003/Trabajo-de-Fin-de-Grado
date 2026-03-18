import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

# Ruta de archivos para el Sujeto 3F
ECG = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\ECG\Subject3F_ECG.py"
EDA = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\EDA\Subject3F_EDA.py"
EMG = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\EMG\Subject3F_EMG.py"
PPG = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\PPG\Subject3F_PPG.py"
RESP = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\RESP\Subject3F_RESP.py"
SKT = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Biopac data\SKT\Subject3F_SKT.py"

# Cargar señales
sECG = np.loadtxt(ECG)
sEDA = np.loadtxt(EDA)
sEMG = np.loadtxt(EMG)
sPPG = np.loadtxt(PPG)
sRESP = np.loadtxt(RESP)
sSKT = np.loadtxt(SKT)

# Frecuencia de muestreo original
fs = 2000

# Diezmado
#no se hace diezmado y = decimate(x,4)

# Nueva frecuencia de muestreo
# no cambia, al no hacer diezmado fs_nueva = fs /4 #pasa a ser 500Hz

# Eje temporal  
tECG = np.arange(len(sECG)) / fs
tEDA = np.arange(len(sEDA)) / fs
tEMG = np.arange(len(sEMG)) / fs
tPPG = np.arange(len(sPPG)) / fs
tRESP = np.arange(len(sRESP)) / fs
tSKT = np.arange(len(sSKT)) / fs

# Representar todas las señales en la misma gráfica
#plt.plot(tECG, sECG, label="Señal de Electrocardiograma")
#plt.plot(tEDA, sEDA, label="Señal de Actividad Electrodérmica")
#plt.plot(tEMG, sEMG, label="Señal de Electromiografía")
#plt.plot(tPPG, sPPG, label="Señal de Fotopletismografía")
#plt.plot(tRESP, sRESP, label="Señal Respiratoria")
#plt.plot(tSKT, sSKT, label="Señal de Temperatura Cutánea")

#plt.title("Sujeto 3F")
#plt.xlabel("Tiempo")
#plt.ylabel("Amplitud")
#plt.legend()
#plt.grid(True)

#plt.show()

# ejemplo de representación por partes
# plt.plot(t[:len(t)//64], y[:len(y)//64])

#ejemplo de representación entera 
# axs[0].plot(tECG, sECG)

# Crear figura y ejes (6 filas, 1 columnas)
fig, axs = plt.subplots(6, 1, figsize=(10, 6))
fig.suptitle("Representación de Gráficas Simultáneas", fontsize=14)

#se ajusta el tamaño de la representación
l=1
# Gráfica 1
axs[0].plot(tECG[:len(tECG)//l], sECG[:len(tECG)//l])
axs[0].set_title("Señal de Electrocardiograma")
axs[0].grid(True)

# Gráfica 2
axs[1].plot(tEDA[:len(tEDA)//l], sEDA[:len(tEDA)//l])
axs[1].set_title("Señal de Actividad Electrodérmica")
axs[1].grid(True)

# Gráfica 3
axs[2].scatter(tEMG[:len(tEMG)//l], sEMG[:len(tEMG)//l])
axs[2].set_title("Señal de Electromiografía")
axs[2].grid(True)

# Gráfica 4
axs[3].plot(tPPG[:len(tPPG)//l], sPPG[:len(tPPG)//l])
axs[3].set_title("Señal de Fotopletismografía")
axs[3].grid(True)

# Gráfica 5
axs[4].plot(tRESP[:len(tRESP)//l], sRESP[:len(tRESP)//l])
axs[4].set_title("Señal Respiratoria")
axs[4].grid(True)

# Gráfica 6
axs[5].plot(tSKT[:len(tSKT)//l], sSKT[:len(tSKT)//l])
axs[5].set_title("Señal de Temperatura Cutánea")
axs[5].grid(True)

plt.subplots_adjust(hspace=0.9)
plt.show()