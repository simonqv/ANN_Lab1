import random

import numpy as np
import matplotlib.pyplot as plt



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
    meanB = [-2.0, 0.0]
    sigmaB = 0.5
    x_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[0])).tolist()
    y_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[1])).tolist()

    classB = [x_B, y_B]

    # Uncomment for scatter plot
    plt.scatter(classA[0], classA[1], label="Class A")
    plt.scatter(classB[0], classB[1], label="Class B")
    plt.legend()

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

    #init_weight_x_r = random.uniform(-1, 1)
    #init_weight_y_r = random.uniform(-1, 1)
    #theta = 1
    init_w = np.random.randn(1, 3)[0]#[init_weight_x_r, init_weight_y_r, theta]

    return x_coord, y_coord, target_p, target_d, init_w


# 3.1.2.1
def f_step(X_input, weights):
    y = weights[0] * X_input[0] + weights[1] * X_input[1] + weights[2] * X_input[2]
    if y > 0: return 1
    return 0


def perceptron_learning(X_input, target, weights, eta):
    # plot original decision boundary
    x_axis = np.linspace(-10, 10, 100)
    descision_boundary = (x_axis * -weights[1]/weights[0]) - weights[2]/weights[0]
    plt.plot(x_axis, descision_boundary, label=f'original decision boundary', marker="x")
    # initialize mean sqare error variables
    error_per_epoch = []
    square_error = 0
    # start perceptron learning
    print(X_input)
    eta = 0.001
    epoch = 20
    for i in range(200*epoch):
        X_pattern = [X_input[0][i % 200], X_input[1][i % 200], X_input[2][i % 200]]
        f = f_step(X_pattern, weights)
        error = target[i % 200] - f
        delta_w = [eta * error * X_pattern[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]
        # calculate mean square error
        # reset count at start of new epoch
        square_error += error**2
        if i%200 == 0:
            mean_square_error = square_error/200
            error_per_epoch.append(mean_square_error)
            square_error = 0
        if i == 200 or i == 200*5 or i == 200*10 or i == 200*15:
            # plot decision boundry
            descision_boundary = (x_axis * -weights[1]/weights[0]) - weights[2]/weights[0]
            plt.plot(x_axis, descision_boundary, label=f'decision boundary for epoch {200*15%200}')
    return weights, error_per_epoch


def delta_learning(X_input, target, weights, eta):
    error_per_epoch = []
    square_error = 0
    eta = 0.001
    for i in range(200*200):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], X_input[2][i % 200]]
        wx = weights[0] * X_in[0] + weights[1] * X_in[1] + weights[2] * X_in[2]
        error = target[i % 200] - wx
        delta_w = [eta * error * X_in[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]

        # plot
        if i % 800 == 0:
            x_axis = np.linspace(-10, 10, 100)
            y_print = x_axis * ((-weights[1]) / weights[0]) - weights[2]/weights[0]
            plt.plot(x_axis, y_print, label=f'epoch {i}')


        # calculate mean square error
        # reset count at start of new epoch
        square_error += error**2
        if i%200 == 0:
            mean_square_error = square_error/200
            error_per_epoch.append(mean_square_error)
            square_error = 0
    return weights, error_per_epoch


# delta learning in batches
def delta_batch_learning(X_input, target, weights, eta):
    error_per_epoch = []

    X_input_np = np.array(X_input)
    target_np = np.array(target).reshape(1, -1)
    weights_np = np.array(weights).reshape(1, -1)
    X_transposed = np.transpose(X_input_np)

    # plot
    x_axis = np.linspace(-4, 4, 100)
    y_print = x_axis * ((-weights_np[0][1]) / weights_np[0][0]) - weights_np[0][2]/weights_np[0][0]
    plt.plot(x_axis, y_print, label="before loop", marker="x")

    eta = 0.01
    for i in range(10):
        #error = target_np - np.dot(weights_np, X_input_np)
        error = target_np - weights_np@X_input_np
        delta_W = eta * error@X_transposed
        print(list(delta_W))
        weights_np = weights_np + delta_W
        # calculate mean square error
        mean_square_error = np.mean(np.square(error))
        error_per_epoch.append(mean_square_error)

        y_print = x_axis * ((-weights_np[0][1]) / weights_np[0][0]) - weights_np[0][2]/weights_np[0][0]

        plt.plot(x_axis, y_print, label=f'epoch {i+1}')
        '''
        if i == 1:
            plt.plot(x_axis, y_print, label="epoch 1")
        if i == 2:
            plt.plot(x_axis, y_print, label="epoch 2")
        if i == 3:
            plt.plot(x_axis, y_print, label="epoch 3")
        if i == 5:
            print("WEIGHTS NP: ", weights_np)
            plt.plot(x_axis, y_print, label="epoch 5")
        if i == 7:
            plt.plot(x_axis, y_print, label="epoch 7")
        if i == 10:
            plt.plot(x_axis, y_print, label="epoch 10")
        if i == 13:
            plt.plot(x_axis, y_print, label="epoch 13")
        '''
        
    return weights_np[0].tolist(), error_per_epoch, y_print


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

    eta = 0.001
    # e = t * fstep(weight tarnspo*X)
    print(weights)
    x_axis = np.linspace(-4, 4, 100)
    y_old = x_axis * ((-weights[1]) / weights[0]) + weights[2]
    if not batch:
        if delta:
            weights, error_delta_learning_seq = delta_learning(X_input, target, weights, eta)
            # Uncomment for mean square error plot
            #plt.plot(np.linspace(0, 200, 200), error_delta_learning_seq, label="delta learning sequential eta = 0.001, epochs = 1000")
        else:
            weights, error_perceptron_learning_seq = perceptron_learning(X_input, target, weights, eta)
            # Uncomment for mean sqare error plot
            #plt.plot(np.linspace(0, 200, 200), error_perceptron_learning_seq, label="perception learning sequential eta = 0.001, epochs = 1000")
    else:
        weights, error_delta_learning_batch, y_print = delta_batch_learning(X_input, target, weights, eta)
        # Uncomment for mean sqare error plot
        #plt.plot(np.linspace(0, 20, 20), error_delta_learning_batch, label="delta learning batch eta = 0.001, epochs = 20")

    # Uncomment for plot of decision boundary
    
    # make orthogonal: orthogonal_vector = [-original_vector[1], original_vector[0]]
    # line = [(0,0), (-weights[1], weights[0])]
    '''
    if bias:
        y_print = x_axis * ((-weights[1]) / weights[0]) 
        print("bias ", weights[2])
        #plt.plot(0, weights[2], marker="x", c="pink")
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
    plt.plot(x_axis, y_old, label="OLD", c="black")
    '''
    
# Perceptron learning vs Delta learning
def task1_1():
    x_values, y_values, target_perceptron, target_delta, init_w = generate_data()

    weights = init_w  # try with random start also
    bias = np.ones(200).tolist()
    X_input = [x_values, y_values, bias]
    eta = 0.001

    # perceptron learning
    weights, error_perceptron_learning_seq = perceptron_learning(X_input, target_perceptron, weights, eta)
    #learning(x, y, target_p, init_w, False)

    # delta learning
    #learning(x, y, target_d, init_w, True)

    #REMOVE LATER Delta learning with batch with bias
    #learning(x, y, target_d, init_w, True, True, True)
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
#task1_2()

