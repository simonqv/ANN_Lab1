import random

import numpy as np
import matplotlib.pyplot as plt

EPOCH = 20


# 3.1.1
def generate_data():
    n = 100
    # mean point in distribution, X-axis and Y-axis means.
    # Class A
    meanA = [1.0, 0.3]
    sigmaA = 0.2
    x_A1 = (np.random.permutation(np.random.normal(size=50) * sigmaA - meanA[0])).tolist()
    x_A2 = (np.random.permutation(np.random.normal(size=50) * sigmaA + meanA[0])).tolist()
    #x_A = np.random.permutation(x_A1 + x_A2)
    y_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[1])).tolist()


    # Class B
    meanB = [0.0, -0.1]
    sigmaB = 0.3
    x_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[0])).tolist()
    y_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[1])).tolist()

    # init weight
    init_weight_x_r = random.uniform(-0.5, 0.5)
    init_weight_y_r = random.uniform(-0.5, 0.5)
    theta = 1
    init_w = [init_weight_x_r, init_weight_y_r, -theta]

    return x_A1+x_A2, y_A, x_B, y_B, init_w
   
def sub_sampling(x_A, y_A, x_B, y_B, sub_sample_A, sub_sample_B):

    s_A = int(sub_sample_A*100)
    s_B = int(sub_sample_B*100)

    # Permutations
    temp1 = list(zip(x_A, y_A))
    random.shuffle(temp1)
    xA, yA = zip(*temp1)

    temp2 = list(zip(x_B, y_B))
    random.shuffle(temp2)
    xB, yB = zip(*temp2)


    # extract subset 
    slice_x_A = xA[:s_A] 
    slice_y_A = yA[:s_A] 
    classA = [slice_x_A, slice_y_A]


    slice_x_B = xB[:s_B] 
    slice_y_B = yB[:s_B] 
    classB = [slice_x_B, slice_y_B]
    

    targets_perceptron = np.ones(s_A).tolist() + np.zeros(s_B).tolist()
    targets_delta = np.ones(s_A).tolist() + (np.ones(s_B) * -1).tolist()

    temp = list(zip(slice_x_A + slice_x_B, slice_y_A + slice_y_B, targets_perceptron, targets_delta))
    random.shuffle(temp)
    x_coord, y_coord, target_p, target_d = zip(*temp)
    x_coord, y_coord, target_p, target_d = list(x_coord), list(y_coord), list(target_p), list(target_d)

    return x_coord, y_coord, target_p, target_d, classA, classB

def special_sampling(x_A, y_A, x_B, y_B, left_A, right_A):

    s_A_left_val = int(left_A*50)
    s_A_right_val = int(right_A*50)
    s_B = 100

    x_A_left = x_A[:50]
    x_A_right = x_A[50:]

    y_A_left = y_A[:50]
    y_A_right = y_A[50:]


    # Permutations
    temp1_left = list(zip(x_A_left, y_A_left))
    random.shuffle(temp1_left)
    xA_left, yA_left = zip(*temp1_left)

    temp1_right = list(zip(x_A_right, y_A_right))
    random.shuffle(temp1_right)
    xA_right, yA_right = zip(*temp1_right)

    temp2 = list(zip(x_B, y_B))
    random.shuffle(temp2)
    xB, yB = zip(*temp2)


    # extract subset 
    slice_x_A = xA_left[:s_A_left_val] + xA_right[:s_A_right_val]
    slice_y_A = yA_left[:s_A_left_val] + yA_right[:s_A_right_val]
    classA = [slice_x_A, slice_y_A]
    print("left ", xA_left[:s_A_left_val], "\n")
    print("right ", xA_right[:s_A_right_val], "\n")
    print(slice_x_A)


    slice_x_B = xB[:s_B] 
    slice_y_B = yB[:s_B] 
    classB = [slice_x_B, slice_y_B]
    

    targets_perceptron = np.ones(s_A_left_val + s_A_right_val).tolist() + np.zeros(s_B).tolist()
    targets_delta = np.ones(s_A_left_val + s_A_right_val).tolist() + (np.ones(s_B) * -1).tolist()

    temp = list(zip(slice_x_A + slice_x_B, slice_y_A + slice_y_B, targets_perceptron, targets_delta))
    random.shuffle(temp)
    x_coord, y_coord, target_p, target_d = zip(*temp)
    x_coord, y_coord, target_p, target_d = list(x_coord), list(y_coord), list(target_p), list(target_d)

    return x_coord, y_coord, target_p, target_d, classA, classB

def generate_overlapping_data():
    n = 100
    # mean point in distribution, X-axis and Y-axis means.
    # Class A
    meanA = [1.0, 1.3]
    sigmaA = 0.7
    x_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[0])).tolist()

    y_A = (np.random.permutation(np.random.normal(size=100) * sigmaA + meanA[1])).tolist()

    classA = [x_A, y_A]

    # Class B
    meanB = [0.0, -0.1]
    sigmaB = 0.6
    x_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[0])).tolist()
    y_B = (np.random.permutation(np.random.normal(size=100) * sigmaB + meanB[1])).tolist()

    classB = [x_B, y_B]
    plt.scatter(classA[0], classA[1], label="Class A")
    plt.scatter(classB[0], classB[1], label="Class B")
 
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
def f_step(X_input, weights):
    y_prim = weights[0] * X_input[0] + weights[1] * X_input[1] + (-weights[2]) * X_input[2]
    if y_prim > 0: return 1
    # If on the edge, count as zero
    return 0


def perceptron_learning(X_input, target, weights, eta):
    for i in range(200*2000):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], X_input[2][i % 200]]
        f = f_step(X_in, weights)
        delta_w = [((-eta) * (target[i % 200] - f)) * X_in[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]

    return weights


def delta_learning(X_input, target, weights, eta):
    len(target)
    for i in range(200*20):
        X_in = [X_input[0][i % 200], X_input[1][i % 200], X_input[2][i % 200]]
        wx = weights[0] * X_in[0] + weights[1] * X_in[1] + weights[2] * X_in[2]
        delta_w = [(-eta * (target[i % 200] - wx)) * X_in[n] for n in range(3)]
        weights = [weights[n] + delta_w[n] for n in range(3)]

    return weights


# delta learning in batches
def batch_learning(X_input, target, weights, eta):
    print(len(X_input))
    print(len(X_input[0]))
    print(len(X_input[1]))
    print(len(X_input[2]))
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

    if bias:
        weights = init_w  # try with random start also
        bias = np.ones(len(x_coord)).tolist()
        X_input = [x_coord, y_coord, bias]

    else:
        weights = init_w[0:2]  # try with random start also
        X_input = [x_coord, y_coord]

    eta = 0.0001
    # e = t * fstep(weight tarnspo*X)
    x_axis = np.linspace(-4, 4, 100)
 
    if not batch:
        if delta:
            weights = delta_learning(X_input, target, weights, eta)
        else:
            weights = perceptron_learning(X_input, target, weights, eta)
    else:
        weights = batch_learning(X_input, target, weights, eta)

    
    y_print = x_axis * ((-weights[1]) / weights[0]) + weights[2]
    return y_print
    

# Data not linearly separable
def task2_1():
    x, y, target_p, target_d = generate_overlapping_data()

    # perceptron learning
    learning(x, y, target_p, False)

    # delta learning
    learning(x, y, target_d, True)
    plt.legend()
    plt.show()

def task2_2():

    fig, axes = plt.subplots(2,3, figsize=(10,6))
    x_axis = np.linspace(-3, 3, 100)
    x_A, y_A, x_B, y_B, init_w = generate_data()
    #plt.scatter(x_A, y_A)
    #plt.scatter(x_B, y_B)
    #plt.show()

    # delta learning, batch and bias with full dataset
    x, y, _, target_d, class_A, class_B = sub_sampling(x_A, y_A, x_B, y_B, 1.0, 1.0)
    y_axis = learning(x, y, target_d, init_w, True, True, True)
   
    axes[0,0].set_title("Delta learning, batch and bias with full dataset")
    axes[0,0].plot(x_axis, y_axis, label="full")
    axes[0,0].scatter(class_A[0], class_A[1], label="A")
    axes[0,0].scatter(class_B[0], class_B[1], label="B")
    axes[0,0].set_ylim(-2, 2)

    # same but remove 25%
    x2, y2, _, target_d2, ca2, cb2 = sub_sampling(x_A, y_A, x_B, y_B, 0.75, 0.75)
    y_axis2 = learning(x2, y2, target_d2, init_w, True, True, True)

    axes[0,1].set_title("Delta learning, batch and bias with 75% of each dataset")
    axes[0,1].plot(x_axis, y_axis2, label="25%")
    axes[0,1].scatter(ca2[0], ca2[1], label="A")
    axes[0,1].scatter(cb2[0], cb2[1], label="B")
    axes[0,1].set_ylim(-2, 2)
  
    
    # delta batch w bias but remove 50% of class A
    x3, y3, _, target_d3, ca3, cb3 = sub_sampling(x_A, y_A, x_B, y_B, 0.5, 1.0)
    y_axis3 = learning(x3, y3, target_d3, init_w, True, True, True)
    
    axes[1,0].set_title("Delta learning, batch and bias with 50% of  class A")
    axes[1,0].plot(x_axis, y_axis3, label="50% class A")
    axes[1,0].scatter(ca3[0], ca3[1], label="A")
    axes[1,0].scatter(cb3[0], cb3[1], label="B")
    axes[1,0].set_ylim(-2, 2)
      

    # delta batch w bias but remove 50% of class B
    x4, y4, _, target_d4, ca4, cb4 = sub_sampling(x_A, y_A, x_B, y_B, 1.0, 0.5)
    y_axis4 = learning(x4, y4, target_d4, init_w, True, True, True)
    axes[1,1].set_title("Delta learning, batch and bias with 50% of  class B")
    axes[1,1].plot(x_axis, y_axis4, label="50% class B")
    axes[1,1].scatter(ca4[0], ca4[1], label="A")
    axes[1,1].scatter(cb4[0], cb4[1], label="B")
    axes[1,1].set_ylim(-2, 2)

    x5, y5, _, target_d5, ca5, cb5 = special_sampling(x_A, y_A, x_B, y_B, 0.2, 0.8)
    y_axis5 = learning(x5, y5, target_d5, init_w, True, True, True)
    axes[1,2].set_title("Delta learning, batch and bias with A divided into 20 and 80")
    axes[1,2].plot(x_axis, y_axis5, label="20% left, 80% right")
    axes[1,2].scatter(ca5[0], ca5[1], label="A")
    axes[1,2].scatter(cb5[0], cb5[1], label="B")
    axes[1,2].set_ylim(-2, 2)

    
    #plt.legend()

    
    plt.show()

'''
      plt.scatter(classA[0], classA[1], label="Class A")
    plt.scatter(classB[0], classB[1], label="Class B")
    plt.legend()
    # plt.show()
'''

#task2_1()
task2_2()