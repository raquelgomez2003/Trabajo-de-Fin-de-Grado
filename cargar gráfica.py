import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración de la ventana
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.geometry("1000x700")
root.title("Visualización de Señal Seleccionada")

# Variables globales
sECG = None
fs = None
canvas = None  # Para poder destruir la gráfica anterior

# Función para cargar el archivo
def cargar_archivo():
    global sECG
    file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"),("Python files", "*.csv")])
    if file_path:
        try:
            sECG = np.loadtxt(file_path)
            messagebox.showinfo("Archivo cargado", f"Se cargó correctamente: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")

# Función para graficar la señal
def graficar():
    global sECG, fs, canvas
    if sECG is None:
        messagebox.showwarning("Atención", "Primero cargue un archivo de señal.")
        return

    try:
        fs_usuario = float(entry_fs.get())
        inicio_min = float(entry_inicio.get())
        fin_min = float(entry_fin.get())
    except:
        messagebox.showwarning("Atención", "Ingrese valores numéricos válidos.")
        return

    # Convertir minutos a índices
    inicio_idx = int(inicio_min * 60 * fs_usuario)
    fin_idx = int(fin_min * 60 * fs_usuario)
    if inicio_idx < 0: inicio_idx = 0
    if fin_idx > len(sECG): fin_idx = len(sECG)
    if inicio_idx >= fin_idx:
        messagebox.showwarning("Atención", "El inicio debe ser menor que el fin.")
        return

    sECG_seccion = sECG[inicio_idx:fin_idx]

    # Diezmado si fs diferente de 2000 Hz
    if fs_usuario != 2000:
        factor = int(fs_usuario / 2000)
        if factor < 1:
            factor = 1
        sECG_seccion = sECG_seccion[::factor]
        fs_final = fs_usuario / factor
    else:
        fs_final = fs_usuario

    tECG = np.arange(len(sECG_seccion)) / fs_final

    # Limpiar gráfica anterior si existe
    if canvas:
        canvas.get_tk_widget().destroy()

    # Crear figura de Matplotlib
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(tECG, sECG_seccion * 1000)  # Convertir a mV
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (mV)")
    ax.set_title("Señal ECG")
    ax.grid(True)

    # Integrar en Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

# Función para eliminar la señal y la gráfica
def eliminar_senal():
    global canvas, sECG
    if canvas:
        canvas.get_tk_widget().destroy()
        canvas = None
    sECG = None
    entry_fs.delete(0, "end")
    entry_inicio.delete(0, "end")
    entry_fin.delete(0, "end")
    messagebox.showinfo("Eliminado", "Señal y gráfica eliminadas. Puede subir un nuevo archivo.")

# Función para cerrar todo el programa sin errores
def cerrar_programa():
    global canvas
    if canvas:
        canvas.get_tk_widget().destroy()
    root.destroy()  # Cierra todo inmediatamente

# Botones y entradas
btn_cargar = ctk.CTkButton(root, text="Cargar Archivo", command=cargar_archivo)
btn_cargar.pack(pady=10)

ctk.CTkLabel(root, text="Frecuencia de muestreo (Hz):").pack()
entry_fs = ctk.CTkEntry(root)
entry_fs.pack(pady=5)

# Entradas para sección en minutos
ctk.CTkLabel(root, text="Sección a representar (minutos):").pack(pady=(10,0))
frame_tiempo = ctk.CTkFrame(root)
frame_tiempo.pack(pady=5)

ctk.CTkLabel(frame_tiempo, text="Inicio:").grid(row=0, column=0, padx=5)
entry_inicio = ctk.CTkEntry(frame_tiempo, width=50)
entry_inicio.grid(row=0, column=1, padx=5)

ctk.CTkLabel(frame_tiempo, text="Fin:").grid(row=0, column=2, padx=5)
entry_fin = ctk.CTkEntry(frame_tiempo, width=50)
entry_fin.grid(row=0, column=3, padx=5)

btn_graficar = ctk.CTkButton(root, text="Graficar Señal", command=graficar)
btn_graficar.pack(pady=10)

btn_eliminar = ctk.CTkButton(root, text="Eliminar Señal/Gráfica", command=eliminar_senal)
btn_eliminar.pack(pady=10)

btn_salir = ctk.CTkButton(root, text="Salir", command=cerrar_programa)
btn_salir.pack(pady=20)

# Asociar la acción de cerrar ventana con nuestra función segura
root.protocol("WM_DELETE_WINDOW", cerrar_programa)

root.mainloop()