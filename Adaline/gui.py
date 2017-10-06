#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import random
import matplotlib
import math
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


matplotlib.use("TkAgg")


class Object:
    def __init__(self, x, y, type):
        self.x = pixels2cartesian(pixel=x)
        self.y = -pixels2cartesian(pixel=y)
        self.type = type


def pixels2cartesian(pixel):
    x = (pixel / 15) - 10
    return x


class Adaline:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.grid()
        self.right_frame = tk.Frame(self.master)

        """Variables para su uso en el programa"""
        self._running = True
        self.end_program = False
        self.example_var = tk.StringVar()
        self.figure = Figure(figsize=(2, 2.5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.width = 300
        self.height = 300
        self.pixelSize = self.width / 20
        self.max_epoch_entry = tk.Entry(self.right_frame, validate="focusout",
                                        vcmd=self.validate_max_epochs)
        self.learning_rate_entry = tk.Entry(self.right_frame, validate="focusout",
                                            vcmd=self.validate_learning_rate)
        self.min_error_sb = tk.Spinbox(self.right_frame, values=(0.0001, 0.001, 0.01, 0.1))
        self.x_proof = tk.Entry(self.right_frame, validate="focusout",
                                vcmd=self.validate_x_proof)
        self.y_proof = tk.Entry(self.right_frame, validate="focusout",
                                vcmd=self.validate_y_proof)

        self.training_canvas = tk.Canvas(self.master, width=300, height=300, background='white')

        self.circles = []
        self.rectangles = []

        self.weights = []

        """Constantes para su uso en el programa"""
        self.GRID_COLOR = '#F3ECEC'
        self.ERROR_COLOR = '#850e04'
        self.DIMENSION = 2

        """Inicializacion y posicionamiento de los widgets"""
        self.make_widgets()

        self.graph_errors = []

    def make_widgets(self):
        # Canvas de entrenamiento
        self.training_canvas.grid(row=0, column=0)
        self.training_canvas.bind("<Button-1>", self.draw_circle)
        self.training_canvas.bind("<Button-3>", self.draw_rectangle)

        self.right_frame.grid(row=0, column=1, sticky='n')

        # CUADRICULA DE 20 * 20
        # Creates all vertical lines at intervals of 15
        for i in range(0, self.width, 15):
            self.training_canvas.create_line([(i, 0), (i, self.height)], tag='grid_line', fill=self.GRID_COLOR)

        # Creates all horizontal lines at intervals of 15
        for i in range(0, self.height, 15):
            self.training_canvas.create_line([(0, i), (self.width, i)], tag='grid_line', fill=self.GRID_COLOR)

        self.training_canvas.create_line(150, 0, 150, 300)
        self.training_canvas.create_line(0, 150, 300, 150)

        # self.init = self.training_canvas.create_line(0, 0, 0, 0, fill="blue", dash=(4, 4))

        # Opciones
        tk.Label(self.right_frame, text="Num. Máx. Épocas: ").grid(column=0, row=0, sticky="w")
        self.max_epoch_entry.grid(column=1, row=0, sticky="we")

        tk.Label(self.right_frame, text="Learning rate: ").grid(column=0, row=1, sticky="w")
        self.learning_rate_entry.grid(column=1, row=1, sticky="we")

        initialize_btn = tk.Button(self.right_frame, text="Inicializar", command=self.clean_everything)
        initialize_btn.grid(column=0, row=3, sticky="we", columnspan=2)

        tk.Label(self.right_frame, text="Error mínimo deseado:").grid(column=0, row=2, sticky="w")
        self.min_error_sb.grid(column=1, row=2, sticky="we")

        tk.Label(self.right_frame, text="X: ").grid(column=2, row=0, sticky="w")
        self.x_proof.grid(column=3, row=0, sticky="we")
        tk.Label(self.right_frame, text="Y: ").grid(column=2, row=1, sticky="w")
        self.y_proof.grid(column=3, row=1, sticky="we")

        proof_btn = tk.Button(self.right_frame, text="Prueba", command=self.proof)
        proof_btn.grid(column=2, row=2, sticky="we", columnspan=2)

        train_btn = tk.Button(self.master, text="Entrenar", command=self.train)
        train_btn.grid(column=0, row=1, sticky="we")

    def draw_circle(self, event):
        self.circles.append(
            self.training_canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='blue'))

    def draw_rectangle(self, event):
        self.rectangles.append(
            self.training_canvas.create_rectangle(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='red'))

    def draw_circle_proof(self, x, y):
        self.circles.append(self.training_canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='blue'))

    def draw_rectangle_proof(self, x, y):
        self.rectangles.append(self.training_canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill='red'))

    def _guess(self, input):
        net = 0
        for i in range(len(self.weights)):
            net += input[i] * self.weights[i]
        return net # self._sign(net)

    def _sign(self, n):
        if n >= 0:
            return 1
        else:
            return -1

    def _logsig(self, y):
        return 1/(1 + math.exp(-y))

    def _draw_line(self):
        x1 = -10
        y1 = (self.weights[2] - (self.weights[0] * x1)) / self.weights[1]
        x2 = 10
        y2 = (self.weights[2] - (self.weights[0] * x2)) / self.weights[1]

        self.perceptron = self.training_canvas.create_line(self.cartesian2pixels(x1), self.cartesian2pixels(-y1),
                                                           self.cartesian2pixels(x2), self.cartesian2pixels(-y2),
                                                           fill="red", dash=(4, 4))

    def fill_inputs(self):
        inputs = []

        for i in range(len(self.circles)):
            inputs.append(Object(self.training_canvas.coords(self.circles[i])[2], self.training_canvas.coords(
                    self.circles[i])[3], 1))
        for i in range(len(self.rectangles)):
            inputs.append(Object(self.training_canvas.coords(self.rectangles[i])[2], self.training_canvas.coords(
                    self.rectangles[i])[3], 0))

        return inputs

    def clean_everything(self):
        for i in range(len(self.circles)):
            self.training_canvas.delete(self.circles[i - 1])
        for i in range(len(self.rectangles)):
            self.training_canvas.delete(self.rectangles[i - 1])
        self.training_canvas.delete(self.perceptron)
        self.circles = []
        self.rectangles = []

    def train(self):
        inputs = self.fill_inputs()
        self.generate_plot()
        self.weights = []
        n = len(self.circles) + len(self.rectangles)
        done = False
        self.graph_errors = []
        error = 200
        e = 0

        learning_rate = float(self.learning_rate_entry.get())
        max_epochs = int(self.max_epoch_entry.get())
        min_wanted_error = float(self.min_error_sb.get())
        self.weights.append(0)
        for i in range(1, self.DIMENSION+1):
            self.weights.append(random.uniform(-1, 1))
        self._draw_line()
        
        while e < max_epochs:
            graph_error = 0
            for j in range(0, n):
                input = [inputs[j].x, inputs[j].y, -1]
                target = inputs[j].type
                guess = self._guess(input)
                error = target - self._logsig(guess)
                graph_error += error ** 2 / 2
                for i in range(len(self.weights)):
                    self.weights[i] += error * input[i] * learning_rate * self._logsig(guess) * \
                                       (1 - self._logsig(guess))
                    self.training_canvas.delete(self.perceptron)
                    self._draw_line()
            self.fig.canvas.draw()
            self.graph_errors.append(graph_error)
            plt.plot(self.graph_errors,  c=self.ERROR_COLOR)

            print('Epocas: ', e)
            print('Error: ', graph_error)
            if e >= max_epochs:
                break;
            e += 1

    def proof(self):
        x = self.cartesian2pixels(int(self.x_proof.get()))
        y = (self.cartesian2pixels(-int(self.y_proof.get())))
        pX = pixels2cartesian(x)
        pY = -pixels2cartesian(y)

        print('X: ', x)
        print('pixel X: ', pX)
        print('Y: ', y)
        print('pixel Y: ', pY)

        input = [x, y, 1]
        guess = self._guess(input)

        if guess == 1:
            self.draw_circle_proof(x, y)
        elif guess == -1:
            self.draw_rectangle_proof(x, y)

        print(guess)

    def generate_plot(self):
        self.fig = plt.figure(1, figsize=(2, 2))
        plt.ion()
        self.error_canvas = FigureCanvasTkAgg(self.fig, self.right_frame)
        self.error_canvas.get_tk_widget().grid(column=0, row=4, columnspan=4, rowspan=3, sticky="wens")

    def validate_max_epochs(self):
        try:
            max_epochs = int(self.max_epoch_entry.get())
        except ValueError:
            messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números enteros.")
            return False
        return True

    def validate_learning_rate(self):
        try:
            learning_rate = float(self.learning_rate_entry.get())
        except ValueError:
            messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números.")
            return False
        return True

    def validate_x_proof(self):
        try:
            x_proof_validate = int(self.x_proof.get())

            if x_proof_validate > 10 or x_proof_validate < -10:
                messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números entre -10 y 10.")
        except ValueError:
            messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números.")
            return False
        return True

    def validate_y_proof(self):
        try:
            y_proof_validate = int(self.y_proof.get())

            if y_proof_validate > 10 or y_proof_validate < -10:
                messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números entre -10 y 10.")
        except ValueError:
            messagebox.showwarning("¡Advertencia!", "Sólo está permitido introducir números.")
            return False
        return True

    def cartesian2pixels(self, x):
        pixel = (x + 10) * 15
        return pixel
