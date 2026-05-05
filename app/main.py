import customtkinter as ctk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from data_loader import cargar_sujeto
from plotter import graficar_multisenal_con_intervalos

# -----------------------------
# CONFIG
# -----------------------------
BASE_PATH = r"C:\Users\raque\Desktop\TFG\base de datos TFG"
fs = 2000

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.geometry("1200x750")
root.title("Visor Biomédico")

senales = {}
canvas = None
toolbar = None

# -----------------------------
# FUNCIONES
# -----------------------------
def cargar_sujeto_ui(num):
    global senales
    try:
        senales = cargar_sujeto(BASE_PATH, num)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return
    graficar()

# -----------------------------
def graficar():
    global canvas, toolbar

    if len(senales) == 0:
        return

    # leer intervalos
    try:
        intervalos = [
            (float(e1.get()), float(e2.get()), 'green'),
            (float(e3.get()), float(e4.get()), 'red')
        ]
    except:
        intervalos = None

    # limpiar anterior
    if canvas:
        canvas.get_tk_widget().destroy()
    if toolbar:
        toolbar.destroy()

    # gráfica
    canvas, fig = graficar_multisenal_con_intervalos(
        senales, fs, frame_grafica, intervalos
    )

    # 🔥 TOOLBAR (EN FRAME SEPARADO)
    toolbar = NavigationToolbar2Tk(canvas, frame_toolbar)
    toolbar.update()
    toolbar.pack(side="left")

# -----------------------------
def reset_vista():
    if toolbar:
        toolbar.home()

# -----------------------------
# INTERFAZ
# -----------------------------

# PANEL IZQUIERDO (SUJETOS)
frame_left = ctk.CTkFrame(root, width=200)
frame_left.pack(side="left", fill="y", padx=10, pady=10)

ctk.CTkLabel(frame_left, text="Sujetos", font=("Arial", 16)).pack(pady=10)

for i in range(1, 6):
    ctk.CTkButton(
        frame_left,
        text=f"Sujeto {i}",
        command=lambda n=i: cargar_sujeto_ui(n)
    ).pack(pady=5, fill="x")

# PANEL SUPERIOR (INPUTS)
frame_top = ctk.CTkFrame(root)
frame_top.pack(side="top", fill="x", padx=10, pady=10)

ctk.CTkLabel(frame_top, text="Calming (inicio-fin)").pack(side="left")
e1 = ctk.CTkEntry(frame_top, width=60)
e1.pack(side="left", padx=5)
e2 = ctk.CTkEntry(frame_top, width=60)
e2.pack(side="left", padx=5)

ctk.CTkLabel(frame_top, text="Stress (inicio-fin)").pack(side="left", padx=10)
e3 = ctk.CTkEntry(frame_top, width=60)
e3.pack(side="left", padx=5)
e4 = ctk.CTkEntry(frame_top, width=60)
e4.pack(side="left", padx=5)

ctk.CTkButton(frame_top, text="Actualizar", command=graficar).pack(side="left", padx=20)
ctk.CTkButton(frame_top, text="Reset Vista", command=reset_vista).pack(side="left")

# 🔥 FRAME TOOLBAR (CLAVE PARA QUE SE VEA)
frame_toolbar = ctk.CTkFrame(root)
frame_toolbar.pack(side="top", fill="x", padx=10)

# PANEL GRÁFICA
frame_grafica = ctk.CTkFrame(root)
frame_grafica.pack(side="right", expand=True, fill="both", padx=10, pady=10)

# CIERRE
root.protocol("WM_DELETE_WINDOW", root.destroy)

root.mainloop()