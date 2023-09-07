import random

import numpy as np
import matplotlib.pyplot as plt

ETA = 0.001
EPOCH = 30


# 3.1.1
def generate_data():
    n = 100
    # mean point in distribution, X-axis and Y-axis means.
    # Class A
    meanA = [2.0, 0.5]
    sigmaA = 0.5
    x_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[0])).tolist()
    y_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[1])).tolist()

    classA = [x_A, y_A]

    # Class B
    meanB = [-2.0+5, -2.0+5]
    sigmaB = 0.5
    x_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[0])).tolist()
    y_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[1])).tolist()

    classB = [x_B, y_B]

    # Uncomment for scatter plot
    plt.scatter(classA[0], classA[1], label="Class A")
    plt.scatter(classB[0], classB[1], label="Class B")
    plt.legend()

    # Permutations
    targets_perceptron = np.ones(100).tolist() + np.zeros(100).tolist()
    targets_delta = np.ones(100).tolist() + (np.ones(100) * -1).tolist()

    temp = list(zip(x_A + x_B, y_A + y_B, targets_perceptron, targets_delta))
    random.shuffle(temp)
    x_coord, y_coord, target_p, target_d = zip(*temp)
    x_coord, y_coord, target_p, target_d = list(x_coord), list(y_coord), list(target_p), list(target_d)

    # making initial weights
    init_w = np.random.randn(1, 3)[0]  # [init_weight_x_r, init_weight_y_r, theta]
    return x_coord, y_coord, target_p, target_d, init_w


# 3.1.2.1
def f_step(X_input, weights):
    y = weights[0] * X_input[0] + weights[1] * X_input[1] + weights[2] * X_input[2]
    if y > 0: return 1
    return 0


def perceptron_learning(X_input, target, weights):
    for i in range(200 * EPOCH):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], 1]
        f = f_step(X_in, weights)
        error = target[i % 200] - f
        delta_w = [ETA * error * X_in[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]

    return weights


def delta_learning(X_input, target, weights):
    error_per_epoch = []
    square_error = 0
    for i in range(200 * 30):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], 1]
        wx = weights[0] * X_in[0] + weights[1] * X_in[1] + weights[2] * X_in[2]
        error = target[i % 200] - wx
        delta_w = [ETA * error * X_in[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]
        """
        # plot
        if i % 800 == 0:
            x_axis = np.linspace(-10, 10, 100)
            y_print = x_axis * ((-weights[1]) / weights[0]) - weights[2]#/weights[0]
            plt.plot(x_axis, y_print, label=f'epoch {i}')
          # y_decision = (-final_weights[0] * x_decision - final_weights[2]) / final_weights[1]


        # calculate mean square error
        # reset count at start of new epoch
        square_error += error**2
        if i%200 == 0:
            mean_square_error = square_error/200
            error_per_epoch.append(mean_square_error)
            square_error = 0

        """

    return weights, error_per_epoch


# delta learning in batches
def delta_batch_learning(X_input, target, weights):
    error_per_epoch = []

    X_input_np = np.array(X_input)
    target_np = np.array(target).reshape(1, -1)
    weights_np = np.array(weights).reshape(1, -1)
    X_transposed = np.transpose(X_input_np)

    for i in range(EPOCH):
        if np.shape(weights_np) == (1, 3):
            weights_np[0, 2] = 1
        error = target_np - (weights_np @ X_input_np)
        delta_W = ETA * (error @ X_transposed) / len(X_input)
        weights_np = weights_np + delta_W

        # calculate mean square error
        # mean_square_error = np.mean(np.square(error))
        # error_per_epoch.append(mean_square_error)

    return weights_np[0].tolist(), error_per_epoch


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

    if not batch:
        if delta:
            weights, error_delta_learning_seq = delta_learning(X_input, target, weights)
            # Uncomment for mean square error plot
            # plt.plot(np.linspace(0, 200, 200), error_delta_learning_seq, label="delta learning sequential eta = 0.001, epochs = 1000")
        else:
            weights = perceptron_learning(X_input, target, weights)
            # Uncomment for mean sqare error plot
            # plt.plot(np.linspace(0, 200, 200), error_perceptron_learning_seq, label="perception learning sequential eta = 0.001, epochs = 1000")
    else:
        weights, error_delta_learning_batch = delta_batch_learning(X_input, target, weights)
        # Uncomment for mean sqare error plot
        # plt.plot(np.linspace(0, 20, 20), error_delta_learning_batch, label="delta learning batch eta = 0.001, epochs = 20")
    return weights


def delta_learning2(x_coord, y_coord, targets, weights, eta, epochs):
    errors = []  # To store errors for each epoch
    # total_error = 0.0
    for i in range(200 * EPOCH):
        x_in = [x_coord[i % 200], y_coord[i % 200], 1]  # Add bias term (1)
        dot_product = np.dot(weights, x_in)
        error = targets[i % 200] - dot_product
        # total_error += error ** 2  # Sum of squared errors
        delta_w = [eta * error * x for x in x_in]
        weights = [weights[j] + delta_w[j] for j in range(len(weights))]
    # errors.append(total_error)
    return weights, errors


def draw_plot(weights, x_decision, label):

    # Plot the decision boundary line using the final weights
    if len(weights) == 3:
        y_decision = (-weights[0] * x_decision - weights[2]) / weights[1]
    else:
        y_decision = (-weights[0] * x_decision) / weights[1]

    plt.plot(x_decision, y_decision, label=label)


# Perceptron learning vs Delta learning
def task1_1():
    x_values, y_values, target_perceptron, target_delta, init_w = generate_data()

    # perceptron learning
    perceptron_weights = learning(x_values, y_values, target_perceptron, init_w, False)

    # delta learning
    delta_learning_weights = learning(x_values, y_values, target_delta, init_w, True)

    # x-values used for plotting
    x_axis = np.linspace(min(x_values), max(x_values), 100)

    draw_plot(perceptron_weights, x_axis, "Perceptron Learning")
    draw_plot(delta_learning_weights, x_axis, "Delta Learning")

    # Show the plot
    plt.legend()
    plt.ylim(-5,5)
    plt.show()


# Delta learning with batch vs online
def task1_2():
    x_values, y_values, _, target_delta, init_w = generate_data()

    # delta learning online
    delta_online_w = learning(x_values, y_values, target_delta, init_w, True, False)

    # delta learning batch
    delta_batch_w = learning(x_values, y_values, target_delta, init_w, True, True)

    # x-values used for plotting
    x_axis = np.linspace(min(x_values), max(x_values), 100)

    draw_plot(delta_online_w, x_axis, "Online Delta")
    draw_plot(delta_batch_w, x_axis, "Batch Delta")

    plt.legend()
    plt.ylim(-5,5)
    plt.show()


# Delta batch learning with bias vs without bias
def task1_3():
    x_values, y_values, _, target_d, init_w = generate_data()

    # Delta learning with batch with bias
    batch_with_bias_w = learning(x_values, y_values, target_d, init_w, True, True, True)

    # Delta learning with batch without bias
    delta_batch_without_w = learning(x_values, y_values, target_d, init_w, True, True, False)

    # x-values used for plotting
    x_axis = np.linspace(min(x_values), max(x_values), 100)

    draw_plot(batch_with_bias_w, x_axis, "Delta with bias")
    draw_plot(delta_batch_without_w, x_axis, "Delta without bias")

    plt.legend()
    plt.ylim(-5,5)
    plt.show()


# Data not linearly separable
def task2_1():
    print("not done")


# task1_1()
# task1_2()
task1_3()