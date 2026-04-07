import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

# Ruta de archivos para el Sujeto 3F
#ACC = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto1\Empatica_data\Subject3F_ACC.csv"
BVP = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto1\Empatica_data\Subject3F_BVP.csv"
EDA = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto1\Empatica_data\Subject3F_EDA.csv"
HR = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto1\Empatica_data\Subject3F_HR.csv"
#IBI = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto1\Empatica_data\Subject3F_IBI.csv"
TEMP = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto1\Empatica_data\Subject3F_TEMP.csv"

# Cargar señales
#sACC = np.loadtxt(ACC)
sBVP = np.loadtxt(BVP)
sEDA = np.loadtxt(EDA)
sHR = np.loadtxt(HR)
#sIBI = np.loadtxt(IBI)
sTEMP = np.loadtxt(TEMP)

# Frecuencia de muestreo original
fs = 4

# Diezmado
#no se hace diezmado y = decimate(x,4)

# Nueva frecuencia de muestreo
# no cambia, al no hacer diezmado fs_nueva = fs /4 #pasa a ser 500Hz

# Eje temporal  
#tACC = np.arange(len(sACC)) / fs
tBVP = np.arange(len(sBVP)) / fs
tEDA = np.arange(len(sEDA)) / fs
tHR = np.arange(len(sHR)) / fs
#tIBI = np.arange(len(sIBI)) / fs
tTEMP = np.arange(len(sTEMP)) / fs

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

# Intervalos (en segundos)
intervals = [
    (0,1498-120-964, 'White', 'No music'),
    (1498-120-964, 1498-120, 'green', 'Calming music'),
    (1498-120, 1498, 'blue', 'Relax'),
    (1498, 1498+964, 'red', 'Vexing music'),
    (1498+964, 1498+964+120, 'blue', 'Relax'),
    (1498+964+120, 2607, 'White', 'No music')
]
# Gráfica 1
#axs[0].plot(tACC[:len(tACC)//l], sACC[:len(tACC)//l])
#axs[0].set_title("Señal de ACC")
#axs[0].grid(True)

# Gráfica 2
axs[1].plot(tBVP[:len(tBVP)//l], sBVP[:len(tBVP)//l])
axs[1].set_title("Señal de BVP")
axs[1].grid(True)

# Gráfica 3
axs[2].scatter(tEDA[:len(tEDA)//l], sEDA[:len(tEDA)//l])
axs[2].set_title("Señal de EDA")
axs[2].grid(True)

# Gráfica 4
axs[3].plot(tHR[:len(tHR)//l], sHR[:len(tHR)//l])
axs[3].set_title("Señal de HR")
axs[3].grid(True)

# Gráfica 5
#axs[4].plot(tIBI[:len(tIBI)//l], sIBI[:len(tIBI)//l])
#axs[4].set_title("Señal de IBI")
#axs[4].grid(True)

# Gráfica 6
axs[5].plot(tTEMP[:len(tTEMP)//l], sTEMP[:len(tTEMP)//l])
axs[5].set_title("Señal de TEMP")
axs[5].grid(True)

# Pintar el fondo
for start, end, color, label in intervals:
    axs[0].axvspan(start, end, color=color, alpha=0.2, label=label)
    axs[1].axvspan(start, end, color=color, alpha=0.2, label=label)
    axs[2].axvspan(start, end, color=color, alpha=0.2, label=label)
    axs[3].axvspan(start, end, color=color, alpha=0.2, label=label)
    axs[4].axvspan(start, end, color=color, alpha=0.2, label=label)
    axs[5].axvspan(start, end, color=color, alpha=0.2, label=label)

# Evitar duplicados en la leyenda
handles, labels = axs[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))


# IMPORTANTE: fijar posición (rápido)
fig.legend(unique.values(), unique.keys(), loc='upper left', bbox_to_anchor=(0.01, 0.99))  # ajuste fino (x, y))

plt.subplots_adjust(hspace=0.9)
plt.show()

#Representamos valor máximo en el eje temporal
#t_max = tECG[-1]
#print("Tiempo máximo (s):", t_max) Resultado = 2607 segundos