import customtkinter as ctk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox

# -----------------------------
# CONFIG
# -----------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.geometry("1100x700")
root.title("Visor Biomédico")

BASE_PATH = r"C:\Users\raque\Desktop\TFG\base de datos TFG"

senales = {}
senal_actual = None
canvas = None

# -----------------------------
# DETECTAR POR NOMBRE
# -----------------------------
def detectar_tipo(nombre):
    nombre = nombre.upper()

    if "ECG" in nombre: return "ECG"
    if "EDA" in nombre: return "EDA"
    if "EMG" in nombre: return "EMG"
    if "PPG" in nombre: return "PPG"
    if "RESP" in nombre: return "RESP"
    if "SKT" in nombre: return "SKT"

    return None

# -----------------------------
# CARGAR SUJETO
# -----------------------------
def cargar_sujeto(num):
    global senales

    senales = {}

    ruta = os.path.join(BASE_PATH, f"Base1_Sujeto{num}", "Biopac data")

    if not os.path.exists(ruta):
        messagebox.showerror("Error", f"No existe:\n{ruta}")
        return

    for archivo in os.listdir(ruta):
        if archivo.endswith(".csv"):
            tipo = detectar_tipo(archivo)

            if tipo:
                ruta_archivo = os.path.join(ruta, archivo)

                try:
                    data = np.loadtxt(ruta_archivo, delimiter=",", skiprows=1)
                    senales[tipo] = data
                except:
                    print(f"Error leyendo {archivo}")

    actualizar_botones_senales()

# -----------------------------
# BOTONES DE SEÑALES
# -----------------------------
def actualizar_botones_senales():
    for widget in frame_senales.winfo_children():
        widget.destroy()

    for tipo in senales.keys():
        btn = ctk.CTkButton(
            frame_senales,
            text=tipo,
            command=lambda t=tipo: seleccionar_senal(t)
        )
        btn.pack(pady=5, fill="x")

# -----------------------------
# SELECCIONAR SEÑAL
# -----------------------------
def seleccionar_senal(tipo):
    global senal_actual
    senal_actual = tipo
    graficar()

# -----------------------------
# GRAFICAR
# -----------------------------
def graficar():
    global canvas

    if senal_actual is None:
        return

    try:
        fs = float(entry_fs.get())
    except:
        messagebox.showwarning("Error", "Frecuencia inválida")
        return

    signal = senales[senal_actual]
    t = np.arange(len(signal)) / fs

    if canvas:
        canvas.get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(t, signal)
    ax.set_title(senal_actual)
    ax.set_xlabel("Tiempo (s)")
    ax.grid()

    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack()

# -----------------------------
# INTERFAZ
# -----------------------------

# PANEL IZQUIERDO (SUJETOS)
frame_izq = ctk.CTkFrame(root, width=200)
frame_izq.pack(side="left", fill="y", padx=10, pady=10)

ctk.CTkLabel(frame_izq, text="Sujetos", font=("Arial", 16)).pack(pady=10)

for i in range(1, 6):
    btn = ctk.CTkButton(
        frame_izq,
        text=f"Sujeto {i}",
        command=lambda n=i: cargar_sujeto(n)
    )
    btn.pack(pady=5, fill="x")

# PANEL CENTRAL (SEÑALES)
frame_senales = ctk.CTkFrame(root, width=200)
frame_senales.pack(side="left", fill="y", padx=10, pady=10)

ctk.CTkLabel(frame_senales, text="Señales", font=("Arial", 16)).pack(pady=10)

# PANEL DERECHO (GRÁFICA)
frame_der = ctk.CTkFrame(root)
frame_der.pack(side="right", expand=True, fill="both", padx=10, pady=10)

ctk.CTkLabel(frame_der, text="Frecuencia de muestreo (Hz)").pack()
entry_fs = ctk.CTkEntry(frame_der)
entry_fs.pack(pady=5)

btn_plot = ctk.CTkButton(frame_der, text="Graficar", command=graficar)
btn_plot.pack(pady=10)

frame_grafica = ctk.CTkFrame(frame_der)
frame_grafica.pack(expand=True, fill="both", pady=10)

# Cerrar
root.protocol("WM_DELETE_WINDOW", root.destroy)

root.mainloop()