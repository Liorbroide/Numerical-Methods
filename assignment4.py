"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""

import numpy as np
import time
import torch




class Assignment4A:
    def __init__(self):
        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        xs = np.linspace(a, b, int(maxtime*100))
        ys = []
        for i in xs:
            y = 0
            for j in range(200): #Considering the noise is normal, I want to average the Y results
                y += f(i)        # of each x value.
            ys.append(y/200)
        ys = np.array(ys)
        matrix = []
        for i in xs:
            l1 = []
            for j in range(d+1):
                l1.append(i**j)
            l1 = list(reversed(l1))
            matrix.append(l1)
        matrix = np.array(matrix)
        matrixT = matrix.transpose()
        mm = torch.matmul(torch.from_numpy(matrixT), torch.from_numpy(matrix)).numpy()
        mm = [list(i) for i in mm]

        def give_identity_mat(N): #Gives the identity matrix of NxN size.
            identity = [[0 for i in range(N)] for i in range(N)]
            for i in range(0, N):
                identity[i][i] = 1
            return identity

        def rowsub(matrix, row1, row2, n):# subtract row1 from row2
            for i in range(len(matrix[row2])):
                matrix[row2][i] -= n * matrix[row1][i]

        def divide(matrix, row, n): #Row division by n
            for i in range(len(matrix[row])):
                matrix[row][i] /= n

        def inverse(matrix1):
            #Builds an inverse matrix, using Gaussian Elimination.
            rows = len(matrix1)
            invMatrix = give_identity_mat(rows)
            for i in range(rows):
                if matrix1[i][i] != 1:
                    factor = matrix1[i][i]
                    divide(matrix1, i, factor)
                    divide(invMatrix, i, factor)
                for j in range(rows):
                    if i != j:
                        if matrix1[j][i] != 0:
                            factor = matrix1[j][i]
                            rowsub(matrix1, i, j, factor)
                            rowsub(invMatrix, i, j, factor)
            return invMatrix

        inverted = np.array(inverse(mm))
        y_dottranspose = torch.matmul(torch.from_numpy(matrixT), torch.from_numpy(ys))
        coef = torch.matmul(torch.from_numpy(inverted), y_dottranspose)


        def least_square(x): #Constructing the polynom, with the degree of 'd', using the coefs retrieved.
            return_val = np.float32(0)
            for i in range(len(coef)):
                return_val += np.float32(coef[i]*(x**(d-i)))
            return return_val
        return least_square








##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)



    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)







if __name__ == "__main__":
    unittest.main()
