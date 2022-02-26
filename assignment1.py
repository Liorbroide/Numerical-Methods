"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
from sampleFunctions import *
import torch
import copy

class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:



        def Thomas(a, b, c, d):

            splines = len(d)  # Number of spline equations
            a1, b1, c1, d1 = map(np.array, (a, b, c, d))  #Applying numpy array objects on each of the given tuples.
            for i in range(1, splines):
                mat = a1[i - 1] / b1[i - 1]
                b1[i] = b1[i] - mat * c1[i - 1]
                d1[i] = d1[i] - mat * d1[i - 1]

            x = b1
            x[-1] = d1[-1] / b1[-1]

            for j in range(splines - 2, -1, -1):
                x[j] = (d1[j] - c1[j] * x[j + 1]) / b1[j]

            return x

        def newton_rhapson(f, df, x0, line):  # Newton Raphson Algorithm as taught in class
            i = 0
            while i < 20 or abs(f(x0)) > line:
                x0 = x0 - f(x0) / df(x0)
                i += 1
            return x0
        # Finding the control points.
        def bezier_coefficients(ps):

            n = len(ps) - 1

            # Buidling coefficence matrix.
            Coef_mat = 4 * np.identity(n)
            np.fill_diagonal(Coef_mat[1:], 1)
            np.fill_diagonal(Coef_mat[:, 1:], 1)
            Coef_mat[0, 0] = 2
            Coef_mat[n - 1, n - 1] = 7
            Coef_mat[n - 1, n - 2] = 2
            # build points vector
            P = [2 * (2 * ps[i] + ps[i + 1]) for i in range(n)]
            P[0] = ps[0] + 2 * ps[1]
            P[n - 1] = 8 * ps[n - 1] + ps[n]

            def abc_retriever(C):#Building A and B vectors.
                a = []
                b = []
                c = []
                for i in range(n - 1):
                    c.append(C[i][i + 1])
                    b.append(C[i][i])
                    a.append(C[i + 1][i])
                b.append(C[n - 1][n - 1])
                return a, b, c

            a, b, c = abc_retriever(Coef_mat)

            A = Thomas(a, b, c, P)
            B = [0] * n
            for i in range(n - 1):
                B[i] = 2 * ps[i + 1] - A[i + 1]
            B[n - 1] = (A[n - 1] + ps[n]) / 2

            return A, B

        # Return one cubic curve for each point
        def get_bezier_cubic(points):
            A, B = bezier_coefficients(points)
            return [[points[i], A[i], B[i], points[i + 1]] for i in range(len(points) - 1)]

        if (n == 1):
            return lambda x: f((a + b) / 2)
        else:
            x_points = np.linspace(a, b, n)
            y_points = np.array([f(x) for x in x_points])
            x_values = get_bezier_cubic(x_points)
            y_values = get_bezier_cubic(y_points)
            bezier_curves = []
            for i in range(len(x_values)):
                p0 = [x_values[i][0], y_values[i][0]]
                p1 = [x_values[i][1], y_values[i][1]]
                p2 = [x_values[i][2], y_values[i][2]]
                p3 = [x_values[i][3], y_values[i][3]]
                bezier_curves.append([p0, p1, p2, p3])
    #Returning the wanted Y value of a given X.

        def get_val(P0, P1, P2, P3, x):
            coefficients = [-P0[0] + 3 * P1[0] - 3 * P2[0] + P3[0], 3 * P0[0] - 6 * P1[0] + 3 * P2[0],
                            -3 * P0[0] + 3 * P1[0], P0[0] - x]
            x_i = 1
            f2 = lambda x: coefficients[0] * x ** 3 + coefficients[1] * x ** 2 + coefficients[2] * x + coefficients[3]
            deriviation = lambda x: 3 * coefficients[0] * x ** 2 + 2 * coefficients[1] * x + coefficients[2]
            i = 0
            root = newton_rhapson(f2, deriviation, x_i, 0.0001)
            while root > 1 or root < 0:
                if i == 20:
                    break
                root = newton_rhapson(f2, deriviation, x_i, 0.0001)
                x_i = x_i / 2
                i += 1

            return (1 - root) ** 3 * P0[1] + 3 * (1 - root) ** 2 * root * P1[1] + 3 * (1 - root) * root ** 2 * P2[1] + root ** 3 * P3[1]

        def spline_func(x):
            first = 0
            last = len(x_points) - 1
            while first <= last:
                median = (first + last) // 2
                if x_points[median] <= x and x <= x_points[median + 1]:
                    return get_val(bezier_curves[median][0], bezier_curves[median][1], bezier_curves[median][2],bezier_curves[median][3], x)
                else:
                    if x < x_points[median + 1]:
                        last = median - 1
                    else:
                        first = median + 1

        return spline_func

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)


    def test_with_poly_restrict(self):

        ass1 = Assignment1()
        a = np.random.randn(5)
        mean_err = 0
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
