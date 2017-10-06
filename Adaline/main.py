import gui as interface

import tkinter as tk
from tkinter import messagebox
import matplotlib
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def main():
    root = tk.Tk()
    root.title("Adaline")
    root.resizable(width=False, height=False)
    root.protocol("WM_DELETE_WINDOW", sys.exit)
    interface.Adaline(root)
    root.mainloop()


if __name__ == '__main__':
    main()
