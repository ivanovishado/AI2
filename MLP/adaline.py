import math
import random
import matplotlib.pyplot as plt


def sigmoid(y):
        return 1/(1 + math.exp(-y))


class Adaline:
    def __init__(self, rectangles, circles, max_epochs, learning_rate, min_wanted_error, gui):

        self.circles = circles
        self.rectangles = rectangles
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.min_wanted_error = min_wanted_error
        self.gui = gui

        """Variables para su uso en el programa"""
        self._running = True
        self.end_program = False

        """Constantes para su uso en el programa"""
        self.DIMENSION = 2

    def _guess(self, input):
        net = 0
        for i in range(len(input)):
            net += input[i] * self.weights[i]
        return net


    def train(self):
        inputs = self.fill_inputs()
        self.gui.generate_plot()
        n = len(self.circles) + len(self.rectangles)
        done = False
        weights = [random.random() for _ in range(self.DIMENSION + 1)]
        graph_errors = []
        error = 1
        e = 0

        self.gui.draw_line(weights)

        while e <= self.max_epochs or error <= self.min_wanted_error:
            graph_error = 0
            for j in range(0, n):
                input = [inputs[j].x, inputs[j].y, -1]
                target = inputs[j].type
                guess = self._guess(input)
                error = target - sigmoid(guess)
                graph_error += error ** 2 / 2
                for i in range(len(weights)):
                    weights[i] += error * input[i] * self.learning_rate * sigmoid(guess) * (1 - sigmoid(guess))
                    self.gui.training_canvas.delete(self.gui.perceptron)
                    self.gui.draw_line(weights)
            self.gui.fig.canvas.draw()
            graph_errors.append(graph_error)
            plt.plot(graph_errors,  c=self.gui.ERROR_COLOR)

            print('Epocas: ', e)
            print('Error: ', graph_error)
            if e >= self.max_epochs or done:
                break
            e += 1
