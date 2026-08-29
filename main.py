import customtkinter as ctk
from tkinterdnd2 import TkinterDnD

from app import MainWindow


class Root(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    root = Root()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
