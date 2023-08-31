import numpy as np
import matplotlib.pyplot as plt


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


generate_data()