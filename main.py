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
    plt.show()



# 3.1.2.1
def perceptron_learning(classA, classB):
    # targets[A,B]
    targets = [np.ones(100), np.zeros(100)]
    eta = 0.01
    e = 1
    xA = [classA[0], classA[1], np.ones(100).tolist()]
    xB = [classB[0], classB[1], np.ones(100).tolist()]
    weights = eta * e * x


def delta_learning():
    # targets[A,B]
    targets = [np.ones(100), (np.ones(100) * -1)]
    eta = 0.01
    e = 1
    x = 1
    weights = eta * e * x

generate_data()