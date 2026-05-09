"""
main.py — BioVisor entry point
Run with: python main.py
"""
from app.gui.app_window import AppWindow

if __name__ == "__main__":
    app = AppWindow()
    app.mainloop()
