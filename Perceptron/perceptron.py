#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import random
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
matplotlib.use("TkAgg")

"""EZ GAME EZ LIFE EZ PERCEPTRON"""


class Object:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type


class Perceptron:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.grid()
        self.right_frame = tk.Frame(self.master)

        """Variables para su uso en el programa"""
        self._running = True
        self.end_program = False
        self.example_var = tk.StringVar()
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.width = 300
        self.height = 300
        self.pixelSize = self.width / 20
        self.max_epoch_entry = tk.Entry(self.right_frame, validate="focusout",
                                        vcmd=self.validate_max_epochs)
        self.max_epochs = 0
        self.learning_rate_entry = tk.Entry(self.right_frame, validate="focusout",
                                            vcmd=self.validate_learning_rate)
        self.learning_rate = 0.0

        self.circles = []
        self.rectangles = []

        self.weights = []

        """Constantes para su uso en el programa"""
        self.GRID_COLOR = '#F3ECEC'

        """Inicializacion y posicionamiento de los widgets"""
        self.make_widgets()

    def make_widgets(self):
        # Canvas de entrenamiento
        self.training_canvas = tk.Canvas(self.master, width=300, height=300, background='white')
        self.training_canvas.grid(row=0, column=0)
        self.training_canvas.bind("<Button-1>", self.draw_circle)
        self.training_canvas.bind("<Button-3>", self.draw_rectangle)

        self.right_frame.grid(row=0, column=1, sticky='n')

        # CUADRICULA DE 20 * 20
        # Creates all vertical lines at intervals of 15
        for i in range(0, self.width, 15):
            self.training_canvas.create_line([(i, 0), (i, self.height)], tag='grid_line',
                                             fill=self.GRID_COLOR)

        # Creates all horizontal lines at intervals of 15
        for i in range(0, self.height, 15):
            self.training_canvas.create_line([(0, i), (self.width, i)], tag='grid_line',
                                             fill=self.GRID_COLOR)

        self.training_canvas.create_line(150, 0, 150, 300)
        self.training_canvas.create_line(0, 150, 300, 150)

        # EJEMPLO DE COMO DEBERIA VERSE LA LINEA DEL ENTENAMIENTO
        self.perceptron = self.training_canvas.create_line(0, 0, 0, 0, fill="red", dash=(4, 4))

        # Opciones
        tk.Label(self.right_frame, text="Num. Máx. Épocas: ").grid(column=0, row=0, sticky="w")
        self.max_epoch_entry.grid(column=1, row=0, sticky="we")

        tk.Label(self.right_frame, text="Learning rate: ").grid(column=0, row=1, sticky="w")
        self.learning_rate_entry.grid(column=1, row=1, sticky="we")

        initialize_btn = tk.Button(self.right_frame, text="Inicializar",
                                   command=self.clean_everything)
        initialize_btn.grid(column=0, row=2, sticky="we", columnspan=2)

        train_btn = tk.Button(self.master, text="Entrenar",
                                    command=self.prepare_training)
        train_btn.grid(column=0, row=1, sticky="we")

        # Grafica de error
        """self.error_canvas = tk.Canvas(self.right_frame, width=300, height=300, background='white')
        self.error_canvas.grid(column=0, row=3, columnspan=2, rowspan=2, sticky="wens")"""
        self.error_canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.error_canvas.get_tk_widget().grid(column=0, row=3, columnspan=2, rowspan=2,
                                               sticky="wens")

    def draw_circle(self, event):
        self.circles.append(
            self.training_canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='blue'))

    def draw_rectangle(self, event):
        self.rectangles.append(self.training_canvas.create_rectangle(
            event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='red'))

    def _guess(self, input):
        sum = 0
        for i in range(len(self.weights)):
            sum += input[i] * self.weights[i]
        return self._sign(sum)

    def _sign(self, n):
        if n >= 0:
            return 1
        else:
            return -1

    def _draw_line(self):
        x1 = 0
        y1 = (-self.weights[2] - self.weights[0] * x1) / self.weights[1]
        x2 = 300
        y2 = (-self.weights[2] - self.weights[0] * x2) / self.weights[1]

        px1 = (x1 / self.pixelSize) - 10
        py1 = (y1 / self.pixelSize) - 10

        px2 = (x2 / self.pixelSize) - 10
        py2 = (y2 / self.pixelSize) - 10

        print(x1, y1, x2, y2)
        print(px1, py1, px2, py2)

        self.perceptron = self.training_canvas.create_line(x1, y1, x2, y2, fill="red", dash=(4, 4))

    def fill_inputs(self):
        inputs = []
        for i in range(len(self.circles)):
            inputs.append(
                Object(self.training_canvas.coords(self.circles[i])[2], self.training_canvas.coords(self.circles[i])[3],
                       1))
        for i in range(len(self.rectangles)):
            inputs.append(Object(self.training_canvas.coords(self.rectangles[i])[2],
                                 self.training_canvas.coords(self.rectangles[i])[3], -1))
        return inputs

    def clean_everything(self):
        for i in range(len(self.circles)):
            self.training_canvas.delete(self.circles[i - 1])
        for i in range(len(self.rectangles)):
            self.training_canvas.delete(self.rectangles[i - 1])
        self.training_canvas.delete(self.perceptron)

    def prepare_training(self):
        inputs = self.fill_inputs()
        self.weights = [0.0, 0.0, 0.0]
        n = len(self.circles) + len(self.rectangles)
        done = False
        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-1, 1)
        self._draw_line()
        e = 0

        while not done or e <= self.max_epochs:
            done = True
            for j in range(0, n):
                input = [inputs[j].x, inputs[j].y, 1]
                target = inputs[j].type
                guess = self._guess(input)
                error = target - guess
                if error != 0:
                    done = False
                    for i in range(len(self.weights)):
                        self.weights[i] += error * input[i] * self.learning_rate
                        self.training_canvas.delete(self.perceptron)
                        self._draw_line()
            e += 1

            if e > self.max_epochs:
                break
            print(e)

    def validate_max_epochs(self):
        try:
            self.max_epochs = int(self.max_epoch_entry.get())
        except ValueError:
            messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números.")
            return False
        return True

    def validate_learning_rate(self):
        try:
            self.learning_rate = float(self.learning_rate_entry.get())
        except ValueError:
            messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números.")
            return False
        return True

    def train(self, input):
            return None


def main():
    root = tk.Tk()
    root.title("Perceptrón")
    root.resizable(width=False, height=False)
    Perceptron(root)
    root.mainloop()


if __name__ == '__main__':
    main()
