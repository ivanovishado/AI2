import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function (used for activation)
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Sigmoid derivative
def d_sigmoid(x):
    return x*(1-x)


class NeuralNetwork:

    def __init__(self, num_input = 0, num_hidden = 0, num_output = 0, gui=0):
        # number of nodes in each layer

        self.input_nodes = num_input
        self.hidden_nodes = num_hidden
        self.output_nodes = num_output
        self.gui = gui

        # weights matrices, they start at random values
        # wih: weights from input to hidden
        # who: weights from hidden to output
        #self.wih = 2 * np.random.rand(self.hidden_nodes, self.input_nodes) - 1
        self.wih = np.random.rand(self.hidden_nodes, self.input_nodes)
        #self.who = 2 * np.random.rand(self.input_nodes, self.output_nodes) - 1
        self.who = np.random.rand(self.input_nodes, self.output_nodes)

    def copy(self, network):
        self.input_nodes = network.input_nodes
        self.hidden_nodes = network.hidden_nodes
        self.output_nodes = network.output_nodes
        self.wih = network.wih
        self.who = network.who

    def train(self, inputs_array, targets_array, max_epochs, learning_rate, min_wanted_error):
        transcurred_epochs = 0
        error = 200

        self.gui.generate_plot()
        graph_errors = []

        # Turn python lists into matrices
        inputs = np.asarray(inputs_array)
        targets = np.asarray(targets_array)

        while transcurred_epochs < max_epochs and error > min_wanted_error:
            graph_error = 0

            # The outputs of the hidden layers pass through sigmoid activation function
            hidden_outputs = sigmoid(np.dot(inputs, self.wih))
            outputs = sigmoid(np.dot(hidden_outputs, self.who))

            # Error is TARGET - OUTPUT
            output_errors = targets - outputs
            graph_error += np.mean(np.abs(output_errors)) ** 2 / 2
            error = np.mean(np.abs(output_errors))

            if (transcurred_epochs % 100) == 0:
                print("Error:" + str(np.mean(np.abs(output_errors))))

            """backpropagation of errors using the chain rule"""
            # calculating the gradient
            gradient_output = output_errors * d_sigmoid(outputs)

            # hidden errors is output error multiplid by weights
            hidden_errors = gradient_output.dot(self.who.T)

            # gradients for next layer
            gradient_hidden = hidden_errors * d_sigmoid(hidden_outputs)

            # change in weights from HIDDEN --> OUTPUT
            hidden_outputs_t = hidden_outputs.T
            self.who += np.dot(hidden_outputs_t, gradient_output) * learning_rate

            # change in weights from INPUT --> HIDDEN
            inputs_t = inputs.T
            self.wih += np.dot(inputs_t, gradient_hidden) * learning_rate

            transcurred_epochs += 1

        # graph error function
        self.gui.fig.canvas.draw()
        graph_errors.append(graph_error)
        plt.plot(graph_errors, c=self.gui.ERROR_COLOR)

    def proof(self, inputs_array):

        # Turn python lists into matrices
        inputs = np.asarray(inputs_array)

        # The outputs of the hidden layers pass through sigmoid activation function
        hidden_outputs = sigmoid(np.dot(inputs, self.wih))
        outputs = sigmoid(np.dot(hidden_outputs, self.who))

        return outputs
