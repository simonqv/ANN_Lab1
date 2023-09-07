import random

import numpy as np
import matplotlib.pyplot as plt


# 3.1.1
def generate_data():
    n = 100
    # mean point in distribution, X-axis and Y-axis means.
    # Class A
    #meanA = [2.0, 0.5]
    #meanA = [2.0, 4.5]
    meanA = [-5.0, 5.0]
    sigmaA = 0.5
    x_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[0])).tolist()
    y_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[1])).tolist()

    classA = [x_A, y_A]

    # Class B
    #meanB = [10.0, 10.0]
    meanB = [-4.0, 2.0]
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

    # init_weight_x_r = random.uniform(-1, 1)
    # init_weight_y_r = random.uniform(-1, 1)
    # theta = 1
    init_w = np.random.randn(1, 3)[0]  # [init_weight_x_r, init_weight_y_r, theta]
    # init_w = np.array([0.5, 0.6, 1])
    return x_coord, y_coord, target_p, target_d, init_w


# 3.1.2.1
def f_step(X_input, weights):
    y = weights[0] * X_input[0] + weights[1] * X_input[1] + weights[2] * X_input[2]
    if y > 0: return 1
    return 0


def perceptron_learning(X_input, target, weights):
    ETA = 0.001
    for i in range(200 * 20):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], 1]
        f = f_step(X_in, weights)
        error = target[i % 200] - f
        delta_w = [ETA * error * X_in[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]

    return weights


def delta_learning(X_input, target, weights, eta):
    error_per_epoch = []
    square_error = 0
    ETA = 0.001
    for i in range(200 * 50):
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

def bld2(X_input, target, weights, eta):
    # Convert input Python arrays to NumPy arrays
    X_input = np.array(X_input)
    target = np.array([target])
    weights = np.array([weights])
    print(target.shape)
    # Ensure that the dimensions of X_input and target are as expected
    assert X_input.shape == (3, 200), "X_input should have dimensions 3x200"
    assert target.shape == (1, 200), "target should have dimensions 1x200"
    assert weights.shape == (1, 3), "weights should have dimensions 1x3"
    eta = 0.001
    epochs = 1
    for epoch in range(epochs):

        print
        # Calculate the predicted values using the current weights
        predicted = np.dot(weights, X_input)
        
        # Calculate the error between predicted and target
        error = target - predicted
        
        # Update the weights using the delta rule in batch mode
        weights += eta * np.dot(error, X_input.T)
    
    return weights.tolist()[0], [0]

# delta learning in batches
def delta_batch_learning(X_input, target, weights, eta):
    error_per_epoch = []
    X_input_np = np.array(X_input)
    target_np = np.array(target)#.reshape(1, -1)
    weights_np = np.array(weights)#.reshape(1, -1)
    X_transposed = np.transpose(X_input_np)
   
    ETA = 0.005
    weight_sum = 0
    for i in range(50):
        wx = np.dot(weights_np, X_input_np) # (1x3*3*200  -  1x200)*200x3 η(Wx − t)xT
        error = target_np - wx
        #print("SHAPE", np.dot(error/200, X_input_np.T).shape)
        delta_W = ETA * np.dot(error, X_input_np.T)/200 # average error per x/y coordinate value
        print("ERROR", error)
        print(wx)
        weights_np += delta_W

        # calculate mean square error
        mean_square_error = np.mean(np.square(error))
        error_per_epoch.append(mean_square_error)

        #y_print = x_axis * ((-weights_np[0][1]) / weights_np[0][0]) - weights_np[0][2] / weights_np[0][0]

        #plt.plot(x_axis, y_print, label=f'epoch {i + 1}')
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
    print(weights_np)
    #return weights_np[0].tolist(), error_per_epoch
    return weights_np.tolist(), error_per_epoch



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
    # y_old = x_axis * ((-weights[1]) / weights[0]) + weights[2]
    if not batch:
        if delta:
            print("DELTA LEARNIN SEQ")
            weights, error_delta_learning_seq = delta_learning(X_input, target, weights, eta)
            # Uncomment for mean square error plot
            # plt.plot(np.linspace(0, 200, 200), error_delta_learning_seq, label="delta learning sequential eta = 0.001, epochs = 1000")
        else:
            print("PERCEPTRON LEARNING")
            weights = perceptron_learning(X_input, target, weights)
            # Uncomment for mean sqare error plot
            # plt.plot(np.linspace(0, 200, 200), error_perceptron_learning_seq, label="perception learning sequential eta = 0.001, epochs = 1000")
    else:
        print("BATCH")
        weights, error_delta_learning_batch = delta_batch_learning(X_input, target, weights, eta)
        print("WEIGHTS : ", weights)
        # Uncomment for mean sqare error plot
        # plt.plot(np.linspace(0, 20, 20), error_delta_learning_batch, label="delta learning batch eta = 0.001, epochs = 20")
    return weights
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


def delta_learning2(x_coord, y_coord, targets, weights, eta, epochs):
    errors = []  # To store errors for each epoch
    # for epoch in range(epochs):
    # total_error = 0.0
    for i in range(200 * 30):
        x_in = [x_coord[i % 200], y_coord[i % 200], 1]  # Add bias term (1)
        dot_product = np.dot(weights, x_in)
        error = targets[i % 200] - dot_product
        # total_error += error ** 2  # Sum of squared errors
        delta_w = [eta * error * x for x in x_in]
        weights = [weights[j] + delta_w[j] for j in range(len(weights))]
        # print(weights)
    # errors.append(total_error)
    return weights, errors


# Perceptron learning vs Delta learning
def task1_1():
    x_values, y_values, target_perceptron, target_delta, init_w = generate_data()
    weights = init_w  # try with random start also
    bias = np.ones(200).tolist()
    X_input = [x_values, y_values, bias]
    eta = 0.001
    x_decision = np.linspace(min(x_values), max(x_values), 100)
    
    # perceptron learning sequential mode
    weights = learning(x_values, y_values, target_perceptron, init_w, False)
    print("perce", weights)
    y_decision = (-weights[0] * x_decision - weights[2]) / weights[1]
    plt.plot(x_decision, y_decision, label="perceptron learning sequential")

    # delta learning sequential mode
    weights = learning(x_values, y_values, target_delta, init_w, True)
    # final_weights, error = delta_learning2(x_values, y_values, target_delta, init_w, 0.001, 30)

    # Plot the decision boundary line using the final weights
    y_decision = (-weights[0] * x_decision - weights[2]) / weights[1]
    print("delta ", weights)
    plt.plot(x_decision, y_decision, label="delta learning sequential")
    print("DELTA SEQ WEIGHTS", weights)
    # delta learning batch mode
    #weights = learning(x_values, y_values, target_delta, init_w, True)
    weights = learning(x_values, y_values, target_delta, init_w, True, True, True)
    y_decision = (-weights[0] * x_decision - weights[2]) / weights[1]
    plt.plot(x_decision, y_decision, label="delta learning batch", c="r")

    plt.legend()
    plt.title("Decision Boundary")

    # Show the plot
    plt.show()
    # REMOVE LATER Delta learning with batch with bias
    # learning(x, y, target_d, init_w, True, True, True)
    # plt.legend()
    # plt.show()


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
# task1_2()


