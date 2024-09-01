import numpy as np
from collections import deque

def sigmoid(x):
    res = 1.0 / (1.0 + np.exp(-x))
    return res


class Network:
    def __init__(self, input_len, arch):
        self.inputs_len = input_len
        self.arch = arch
        self.rng = np.random.default_rng()
        self.weights = deque()

        self.num_iter = 50_000
        self.eps = 0.1   # h value for derivative
        self.rate = 0.1  # learning rate

    def _cost(self, a, b):
        return sum((a - b)**2)
    
    def cost(self, train_in, train_out):
        total, n = 0, len(train_in)
        for i, o in zip(train_in, train_out):
            pred = self.forward(i)
            total += self._cost(pred, o)
        return total / n


    def train(self, train_in, train_out):
        assert len(train_in) == len(train_out)
        ds = [
            np.zeros(w.shape) for w in self.weights
        ]

        c0 = self.cost(train_in, train_out)
        for e in range(self.num_iter):
            for idx, w in enumerate(self.weights):
                r, c = w.shape
                for i in range(r):
                    for j in range(c):
                        old = w[i][j]
                        w[i][j] += self.eps
                        d = (self.cost(train_in, train_out) - c0) / self.eps
                        ds[idx][i][j] = d[0]
                        w[i][j] = old

            for i in range(len(self.weights)):
                self.weights[i] -= self.rate*ds[i]

            c0 = self.cost(train_in, train_out)
            print("cost: ", c0) 

    # initialise random weights and biases
    def randomise(self):
        num_cols = self.inputs_len
        for num_rows in self.arch:
            weight = self.rng.random((num_rows, num_cols + 1))
            self.weights.appendleft(weight)
            num_cols = weight.shape[0]

    # perform single forward pass
    # basically just left-multiply all the weight matrices
    def forward(self, inputs):
        res = np.vstack((inputs, [1.0]))
        for i in range(len(self.weights)-1, -1, -1):
            res = sigmoid(np.vstack((self.weights[i] @ res, [1.0])) if i != 0 else self.weights[i] @ res)
        return res

def main():
    n = Network(2, [2, 1])
    n.randomise()
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
    n.train(xor_in, xor_out)
    for i in xor_in:
        print(f"{i[0]}^{i[1]} = {n.forward(i)}")
    
if __name__ == "__main__":
    main()
