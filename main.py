import numpy as np
import matplotlib.pyplot as pl


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
    pl.scatter(classA[0], classA[1], c="r", label="class A")
    pl.scatter(classB[0], classB[1], c="b", label="class B")
    pl.legend()



generate_data()