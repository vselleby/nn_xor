import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self):
        self.input_layer_size = 2
        self.hidden_layer_size = 2
        self.output_layer_size = 1

        self.w_hidden = np.random.random((self.input_layer_size, self.hidden_layer_size))
        self.w_output = np.random.random((self.hidden_layer_size, self.output_layer_size))

        self.z_hidden = np.array(0)
        self.a_hidden = np.array(0)
        self.z_output = np.array(0)
        self.a_output = np.array(0)

        self.activation_function = self.sigmoid

    def sigmoid(self, z, derivative=False):
        if derivative:
            return np.exp(-z) / ((1 + np.exp(-z))**2)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z_hidden = np.dot(X, self.w_hidden)
        self.a_hidden = self.activation_function(self.z_hidden)      # This is the output from the Hidden layer

        self.z_output = np.dot(self.a_hidden, self.w_output)
        self.a_output = self.activation_function(self.z_output)      # This is the output from output layer.

        return self.a_output

    def cost_function(self, y):
        return 0.5 * sum((y-self.a_output) ** 2)   # MSE

    def backpropagation(self, X, y):
        delta_a_output = np.multiply(-(y - self.a_output), self.activation_function(self.z_output, derivative=True))
        delta_output_w = np.dot(self.a_hidden.T, delta_a_output)

        delta_a_hidden = np.dot(delta_a_output, self.w_output.T) * self.activation_function(self.z_hidden, derivative=True)
        delta_hidden_w = np.dot(X.T, delta_a_hidden)
        return delta_output_w, delta_hidden_w


if __name__ == "__main__":
    epochs = 10000
    learning_rate = 1

    nn = NeuralNetwork()

    X = np.array([
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1]
    ])

    y = np.array([
        [1],
        [0],
        [1],
        [0]
    ])
    cost = []
    for epoch in xrange(1, epochs):
        nn.forward(X)
        cost.append(nn.cost_function(y))
        output_error, hidden_error = nn.backpropagation(X, y)

        nn.w_output = nn.w_output - (learning_rate * output_error)
        nn.w_hidden = nn.w_hidden - (learning_rate * hidden_error)

    # test = np.array([0.3, 0])
    test = X
    nn.forward(test)
    print(str(test) + '\n\n' + str(nn.w_hidden) + '\n\n' + str(nn.z_hidden) + '\n\n' + str(nn.a_hidden) + '\n\n' +
          str(nn.w_output) + '\n\n' + str(nn.z_output) + '\n\n' + str(nn.a_output))
    plt.plot(np.arange(1, epochs, 1), np.array(cost))
    # plt.show()


