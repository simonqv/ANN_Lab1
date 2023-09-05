import random

import numpy as np
import matplotlib.pyplot as plt

EPOCH = 20


# 3.1.1
def generate_data():
    n = 100
    # mean point in distribution, X-axis and Y-axis means.
    # Class A
    meanA = [3.0, 3.0]
    sigmaA = 0.2
    x_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[0])).tolist()
    y_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[1])).tolist()

    classA = [x_A, y_A]

    # Class B
    meanB = [10.0, 10.0]
    sigmaB = 0.2
    x_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[0])).tolist()
    y_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[1])).tolist()

    classB = [x_B, y_B]
    plt.scatter(classA[0], classA[1], label="Class A")
    plt.scatter(classB[0], classB[1], label="Class B")
    plt.legend()
    # plt.show()

    # Permutations
    x = np.random.permutation(x_A + x_B)
    y = np.random.permutation(y_A + y_B)

    targets_perceptron = np.ones(100).tolist() + np.zeros(100).tolist()
    targets_delta = np.ones(100).tolist() + (np.ones(100) * -1).tolist()

    temp = list(zip(x_A + x_B, y_A + y_B, targets_perceptron, targets_delta))
    random.shuffle(temp)
    x_coord, y_coord, target_p, target_d = zip(*temp)
    x_coord, y_coord, target_p, target_d = list(x_coord), list(y_coord), list(target_p), list(target_d)

    # making initial weights
    init_weight_x_r = random.uniform(-1, 1)
    init_weight_y_r = random.uniform(-1, 1)
    theta = 1
    init_w = [init_weight_x_r, init_weight_y_r, theta]

    return x_coord, y_coord, target_p, target_d, init_w


# 3.1.2.1
def f_step(X_input, weights):
    y_prim = (weights[0] * X_input[0]) + (weights[1] * X_input[1]) + (-weights[2] * X_input[2])
    if y_prim > 0: return 1
    # If on the edge, count as zero
    return 0


def perceptron_learning(X_input, target, weights, eta):
    for i in range(200*2000):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], X_input[2][i % 200]]
        f = f_step(X_in, weights)
        delta_w = [((eta) * (target[i % 200] - f)) * X_in[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]
    return weights


def delta_learning(X_input, target, weights, eta):
    len(target)
    for i in range(200*800):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], X_input[2][i % 200]]
        wx = weights[0] * X_in[0] + weights[1] * X_in[1] + weights[2] * X_in[2]
        delta_w = [eta * ((target[i % 200] - wx) * X_in[n]) for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]

    return weights


# delta learning in batches
def batch_learning(X_input, target, weights, eta):
    X_input_np = np.array(X_input)
    target_np = np.array(target).reshape(1, -1)
    weights_np = np.array(weights).reshape(1, -1)
    X_transposed = np.transpose(X_input_np)

    for i in range(EPOCH):
        # Delta_W = (WX - T)X'
        wx = np.dot(weights_np, X_input_np) - target_np
        delta_W = (eta * np.dot(wx, X_transposed))
        weights_np = weights_np + delta_W

    return weights_np[0].tolist()


def learning(x_coord, y_coord, target, init_w, delta=True, batch=False, bias=True):
    """
    Prepares and runs the learning algorithms
    x_coord:    the x-coordinates of the input data
    y_coord:    the y-coordinates of the input data
    target:     {0,1} for perceptron learning and [-1,1] for delta learning
    delta:      True if delta learning, False if perceptron learning
    batch:      True if batch learning, False otherwise
    bias:       True if bias is used, False otherwise
    """

    # start weights
    
    if bias:
        weights = init_w  # try with random start also
        bias = np.ones(200).tolist()
        X_input = [x_coord, y_coord, bias]

    else:
        weights = init_w[0:2]  # try with random start also
        X_input = [x_coord, y_coord]

    eta = 0.0001
    # e = t * fstep(weight tarnspo*X)
    print(weights)
    x_axis = np.linspace(-4, 4, 100)
    y_old = x_axis * ((-weights[1]) / weights[0]) + weights[2]
    if not batch:
        if delta:
            weights = delta_learning(X_input, target, weights, eta)
        else:
            weights = perceptron_learning(X_input, target, weights, eta)
    else:
        weights = batch_learning(X_input, target, weights, eta)
    print(weights)

    # make orthogonal: orthogonal_vector = [-original_vector[1], original_vector[0]]
    # line = [(0,0), (-weights[1], weights[0])]
    if bias:
        y_print = x_axis * ((-weights[1]) / weights[0]) + weights[2]
        print("bias ", weights[2])
        plt.plot(0, weights[2], marker="x", c="pink")
    else:
        y_print = x_axis * ((-weights[1]) / weights[0])
        plt.plot(0, 0, marker="o", c="r")
    if delta:
        if batch:
            if bias:
                plt.plot(x_axis, y_print, label="delta batch", c="b")
            else:
                plt.plot(x_axis, y_print, label="delta batch without bias", c="g")
        else:
            plt.plot(x_axis, y_print, label="delta", c="r")
    else:
        plt.plot(x_axis, y_print, label="perceptron", c="orange")
    plt.plot(x_axis, y_old, label="OLD")

# Perceptron learning vs Delta learning
def task1_1():
    x, y, target_p, target_d, init_w = generate_data()

    # perceptron learning
    learning(x, y, target_p, init_w, False)

    # delta learning
    learning(x, y, target_d, init_w, True)
    plt.legend()
    plt.show()


# Delta learning with batch vs online
def task1_2():
    x, y, target_p, target_d, init_w = generate_data()

    # delta learning online
    learning(x, y, target_d, init_w, True, False)

    # delta learning batch
    learning(x, y, target_d, init_w, True, True)

    plt.legend()
    plt.show()


# Delta batch learning with bias vs without bias
def task1_3():
    x, y, target_p, target_d, init_w = generate_data()

    # Delta learning with batch with bias
    learning(x, y, target_d, init_w, True, True, True)

    # Delta learning with batch without bias
    learning(x, y, target_d, init_w, True, True, False)

    plt.legend()
    plt.show()


# Data not linearly separable
def task2_1():
    print("not done")


task1_1()

