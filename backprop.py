import numpy as np
from collections import deque
import random
import math
from matplotlib import pyplot as plt
from matplotlib import animation


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Network:
    def __init__(self, inputs_len, arch, num_iter):
        self.inputs_len = inputs_len
        self.arch = arch
        self.rng = np.random.default_rng()
        self.weights = []
        self.biases = []
        self.activations = [
            None for i in range(len(arch))
        ]  # cached activations for inner layers

        self.num_iter = num_iter
        self.rate = 0.1  # learning rate
        self.errors = [None for i in range(len(arch))]
        self.epochs = []  # used for plotting
        self.costs = []  # used for plotting

    def cost(self, y_pred, y):
        return sum(1 / 2 * (y_pred - y) ** 2)

    def make_batches(
        self, train_in, train_out, num_batches
    ) -> list[tuple[np.array, np.array]]:
        batch_size = len(train_in) // num_batches
        start = 0
        batches = []
        for b in range(1, num_batches + 1):
            if b == num_batches:
                batch_in, batch_out = train_in[start:], train_out[start:]
            else:
                end = b * batch_size
                batch_in, batch_out = train_in[start:end], train_out[start:end]
                start = end
            batches.append((batch_in, batch_out))
        return batches

    def process_batch(
        self, batch
    ) -> (list[list[np.array]], list[list[np.array]], np.float64):
        all_activations: list[
            list[np.array]
        ] = []  # 2d matrix of activations per training sample
        all_errors: list[list[np.array]] = []  # 2d matrix of errors per training sample
        total_cost_of_batch = 0
        batch_in, batch_out = batch
        num_samples_in_batch = len(batch_in)
        for i in range(num_samples_in_batch):
            activations = self.forward(batch_in[i])
            all_activations.append(activations)
            errors = self.backward(batch_out[i], activations)
            all_errors.append(errors)
            local_cost = self.cost(activations[-1], batch_out[i])
            total_cost_of_batch += local_cost
        return (all_activations, all_errors, total_cost_of_batch[0])

    def update_params(
        self, all_activations, all_errors, train_in, num_samples_in_batch
    ):
        num_updates = len(self.weights)
        for i in range(num_updates):
            if i != 0:
                delta_w = sum(
                    [
                        errors[i] @ activations[i - 1].transpose()
                        for activations, errors in zip(all_activations, all_errors)
                    ]
                )
                self.weights[i] -= self.rate / num_samples_in_batch * delta_w
            else:
                delta_w = sum(
                    [
                        errors[i] @ sample.transpose()
                        for sample, errors in zip(train_in, all_errors)
                    ]
                )
                self.weights[i] -= self.rate / num_samples_in_batch * delta_w
            delta_b = sum([errors[i] for errors in all_errors])
            self.biases[i] -= self.rate / num_samples_in_batch * delta_b

    def train(self, train_in, train_out):
        num_samples_total = len(train_in)
        batches = self.make_batches(train_in, train_out, 3)
        for i in range(self.num_iter):
            total_cost = 0
            for batch in batches:
                all_activations, all_errors, total_cost_of_batch = self.process_batch(
                    batch
                )
                total_cost += total_cost_of_batch
                self.update_params(all_activations, all_errors, train_in, len(batch))
            if i % 1000 == 0:
                print(f"cost at i={i}: {total_cost / num_samples_total}")
            if i % 100 == 0:
                self.epochs.append(i)
                self.costs.append(total_cost / num_samples_total)

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
    def forward(self, inputs) -> list[np.array]:
        activations = []
        activation = inputs
        n = len(self.weights)
        for i in range(n):
            activation = sigmoid(self.weights[i] @ activation + self.biases[i])
            activations.append(activation)
        return activations

    def backward(self, output, activations) -> list[np.array]:
        # error of output layer
        err_out = activations[-1] * (1 - activations[-1]) * (activations[-1] - output)
        n = len(activations)
        errors = [np.array([]) for i in range(n)]
        errors[-1] = np.array(err_out)

        # error of hidden layers
        for i in range(n - 2, -1, -1):
            err_hidden = (
                activations[i]
                * (1 - activations[i])
                * (self.weights[i + 1].transpose() @ errors[i + 1])
            )
            errors[i] = err_hidden
        return errors

    def graph(self, filename):
        fig, ax = plt.subplots()
        line = ax.plot(self.epochs, self.costs)[0]
        ax.set(xlim=[0, self.num_iter], ylim=[0, 1], xlabel="epoch", ylabel="cost")

        def update(frame):
            # update the line plot:
            line.set_xdata(self.epochs[:frame])
            line.set_ydata(self.costs[:frame])
            ax.set_title(f"epoch: {self.epochs[frame]}, cost: {self.costs[frame]:.5f}")
            return (line,)

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=int(self.num_iter / 100), interval=20
        )
        ani.save(filename)
        plt.show()

    def eval(self, test_in, test_out):
        for idx, sample in enumerate(test_in):
            activation = sample
            n = len(self.weights)
            for i in range(n):
                activation = sigmoid(self.weights[i] @ activation + self.biases[i])
            print(
                int("".join([str(round(element[0])) for element in sample[:4]]), 2),
                end="+ ",
            )
            print(
                int("".join([str(round(element[0])) for element in sample[4:]]), 2),
                end="= ",
            )
            print(int("".join([str(round(element[0])) for element in activation]), 2))


DATA_IN = [
    np.array(
        [[int(b)] for b in format(i, "04b")] + [[int(b)] for b in format(j, "04b")],
        dtype=np.float64,
    )
    for i in range(8)
    for j in range(8)
]
DATA_OUT = [
    np.array([[int(b)] for b in format(i + j, "04b")], dtype=np.float64)
    for i in range(8)
    for j in range(8)
]
DATA_ZIP = list(zip(DATA_IN, DATA_OUT))
random.shuffle(DATA_ZIP)
DATA_IN, DATA_OUT = zip(*DATA_ZIP)

TRAIN_IN, TRAIN_OUT = DATA_IN[:45], DATA_OUT[:45]

nn = Network(8, [12, 12, 4], 100_000)
nn.randomise()
nn.train(TRAIN_IN, TRAIN_OUT)
# nn.graph("./anim.gif")
# nn.eval(TRAIN_IN, TRAIN_OUT)
TEST_IN, TEST_OUT = DATA_IN[45:], DATA_OUT[45:]
