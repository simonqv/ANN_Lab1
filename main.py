import random

import numpy as np
import matplotlib.pyplot as plt


# 3.1.1
def generate_data():
    n = 100
    # mean point in distribution, X-axis and Y-axis means.
    # Class A
    meanA = [1.0, 0.5]
    sigmaA = 0.4
    x_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[0])).tolist()
    y_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[1])).tolist()

    classA = [x_A, y_A]

    # Class B
    meanB = [-2.0, -0.5]
    sigmaB = 0.4
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

    return x_coord, y_coord, target_p, target_d

# 3.1.2.1
def f_step(X_input, weigths):
    y_prim = weigths[0] * X_input[0] + weigths[1] * X_input[1] + weigths[2] * X_input[2]
    if y_prim > 0: return 1
    # If on the edge, count as zero
    return 0

def perceptron_learning(x_coord, y_coord, target):
    # start weights
    weights = [0.5, 0.5, -1]
    bias = np.ones(200).tolist()
    X_input = [x_coord, y_coord, bias]
    eta = 0.01
    # e = t * fstep(weight tarnspo*X)
    print(weights)
    x_axis = np.linspace(-4, 4, 100)
    y_old =  x_axis * ((-weights[1]) / weights[0]) + weights[2]
    for i in range(20000):
        X_in = [X_input[0][i%200], X_input[1][i%200], X_input[2][i%200]]
        f = f_step(X_in, weights)
        new_w = [(eta * (target[i%200] - f)) * X_in[n] for n in range(3)]
        weights = [weights[n] + new_w[n] for n in range(3)]
    print(weights)

    # make orthogonal: orthogonal_vector = [-original_vector[1], original_vector[0]]
    # line = [(0,0), (-weights[1], weights[0])]
    y_print = x_axis * ((-weights[1]) / weights[0]) + weights[2]
    plt.plot(x_axis, y_print, c="r")
    plt.plot(x_axis, y_old, c="b")
    plt.show()

def delta_learning(x_coord, y_coord, target):
    # targets[A,B]
    targets = [np.ones(100), (np.ones(100) * -1)]
    eta = 0.01
    e = 1
    x = 1
    weights = eta * e * x


x, y, target_p, target_d = generate_data()
perceptron_learning(x, y, target_p)
