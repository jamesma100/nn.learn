import math
import random
import time

"""
x^y = (x or y) and not (x and y)

x---
   |
   n00 ----------------
   |                  |
y---                  |
                     n21
x---                  |
   |                  |
   n01 ----- n11 -----|
   |
y---

0 | 0 -> 0
0 | 1 -> 1
1 | 0 -> 1
1 | 1 -> 0

weight labeling: (layer, neuron, input)
"""

TRAIN = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]
EPS = 0.1
RATE = 0.1
NUM_ITER = 100000

def sigmoid(x):
    res = 1.0 / (1.0 + math.exp(-x))
    return res
"""
weights = [w000, w001, w010, w011, w101, w200, w201]
biases = [b00, b01, b10, b20]
"""

def eval(weights, biases, x0, x1):
    n00 = sigmoid(x0 * weights[0] + x1 * weights[1] + biases[0])
    n01 = sigmoid(x0 * weights[2] + x1 * weights[3] + biases[1])
    n11 = sigmoid(n01 * weights[4] + biases[2])
    n21 = sigmoid(n00 * weights[5] + n11 * weights[6] + biases[3])
    return n21

def cost(weights, biases):
    total = 0
    for x0, x1, y in TRAIN:
        yp = eval(weights, biases, x0, x1)
        total += (yp - y)**2
        
    return math.sqrt(total)


def forward(weights, biases):
    c0 = cost(weights, biases)
    dws = []
    dbs = []
    for i, weight in enumerate(weights):
        weights_copy = weights.copy()
        weights_copy[i] += EPS
        dw = (cost(weights_copy, biases) - c0) / EPS
        dws.append(dw)

    for i, bias in enumerate(biases):
        biases_copy = biases.copy()
        biases_copy[i] += EPS
        db = (cost(weights, biases_copy) - c0) / EPS
        dbs.append(db)

    for i in range(len(weights)):
        weights[i] -= RATE * dws[i]

    for i in range(len(biases)):
        biases[i] -= RATE * dbs[i]
    c1 = cost(weights, biases)
    return c1

def main():
    random.seed(time.time())
    weights = [random.random() for i in range(7)]
    biases = [random.random() for i in range(4)]

    print(f"i: -1, cost: {cost(weights, biases)}")
    for i in range(NUM_ITER):
        cost_next = forward(weights, biases)
        print(f"i: {i}, cost: {cost_next}")

    for x0, x1, _ in TRAIN:
        print(f"{x0}^{x1} => {eval(weights, biases, x0, x1)}")

    return 0


if __name__ == "__main__":
    main()
