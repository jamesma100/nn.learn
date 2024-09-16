import numpy as np
from collections import deque

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class Network:
    def __init__(self, input_len, arch, num_iter):
        self.inputs_len = input_len
        self.arch = arch
        self.rng = np.random.default_rng()
        self.weights = deque()
        self.biases = deque()
        self.activations = [None for i in range(len(arch))] # cached activations for inner layers

        self.num_iter = num_iter
        self.rate = 0.1  # learning rate
        self.errors = [None for i in range(len(arch))]


    def cost(self, y):
        return sum((self.activations[-1] - y)**2)

    def train(self, train_in, train_out):
        assert len(train_in) == len(train_out)
        assert len(self.weights) == len(self.errors) == len(self.biases)
        num_samples = len(train_in)
        num_updates = len(self.weights)

        for i in range(self.num_iter):
            total = 0
            for j in range(num_samples):
                self.forward(train_in[j])
                self.backward(train_out[j])
                for k in range(num_updates):
                    if k != 0:
                        self.weights[k] -= (self.rate * self.errors[k]) @ self.activations[k-1].transpose()
                    else:
                        self.weights[k] -= (self.rate * self.errors[k]) @ train_in[j].transpose()
                    self.biases[k] -= (self.rate * self.errors[k])
                total += self.cost(train_out[j])
            if i % 100 == 0:
                print(f"i={i}, cost={total / num_samples}")

    # initialise random weights and biases
    def randomise(self):
        num_cols = self.inputs_len
        for num_rows in self.arch:
            weight = self.rng.random((num_rows, num_cols))
            bias = self.rng.random((num_rows, 1))
            self.weights.append(weight)
            self.biases.append(bias)
            num_cols = weight.shape[0]


    # perform single forward pass
    # basically just left-multiply all the weight matrices
    def forward(self, inputs):
        activation = inputs
        n = len(self.weights)
        for i in range(n):
            activation = sigmoid(self.weights[i]@activation + self.biases[i])
            self.activations[i] = activation

    def backward(self, output):
        # error of output layer
        err_out = self.activations[-1] * (1 - self.activations[-1]) * (self.activations[-1] - output)
        self.errors[-1] = np.array(err_out)

        # error of hidden layers
        n = len(self.activations)
        for i in range(n-2, -1, -1):
            err_hidden = self.activations[i]*(1-self.activations[i])*(self.weights[i+1].transpose() @ self.errors[i+1])
            self.errors[i] = err_hidden
        return err_out

xor_in = [
    np.array([[0], [0]], dtype=np.float64),
    np.array([[0], [1]], dtype=np.float64),
    np.array([[1], [0]], dtype=np.float64),
    np.array([[1], [1]], dtype=np.float64),
]
xor_out = [
    np.array([0], dtype=np.float64),
    np.array([1], dtype=np.float64),
    np.array([1], dtype=np.float64),
    np.array([0], dtype=np.float64),
]

nn = Network(2, [2, 2, 1], 100_000)
nn.randomise()
nn.train(xor_in, xor_out)

for sample in xor_in:
    activation = sample
    n = len(nn.weights)
    for i in range(n):
        activation = sigmoid(nn.weights[i]@activation + nn.biases[i])
    print(f"{sample[0]} ^ {sample[1]} = {activation}")
