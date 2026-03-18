import customtkinter as ctk

# Configuración de la ventana
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()  # ventana principal
root.geometry("300x200")
root.title("Hello World")

label = ctk.CTkLabel(root, text="Hello World")
label.pack(pady=50)

root.mainloop()