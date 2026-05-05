def graficar_multisenal_con_intervalos(senales, fs, frame, intervalos=None):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    fig, axs = plt.subplots(len(senales), 1, figsize=(8, 2*len(senales)), sharex=True)

    if len(senales) == 1:
        axs = [axs]

    for i, (nombre, signal) in enumerate(senales.items()):

        t = np.arange(len(signal)) / fs
        axs[i].plot(t, signal)
        axs[i].set_title(nombre)
        axs[i].grid()

        if intervalos:
            for start, end, color in intervalos:
                axs[i].axvspan(start, end, color=color, alpha=0.2)

    axs[-1].set_xlabel("Tiempo (s)")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()

    widget = canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)

    return canvas, fig   # 🔥 IMPORTANTE