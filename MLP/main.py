import gui as interface
import tkinter as tk
import matplotlib
import sys
matplotlib.use("TkAgg")


def main():
    root = tk.Tk()
    root.title("Multi-layer Network")
    root.resizable(width=False, height=False)
    root.protocol("WM_DELETE_WINDOW", sys.exit)
    interface.AppWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
