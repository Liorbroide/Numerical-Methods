"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from sampleFunctions import *
from sklearn.cluster import KMeans
from functionUtils import AbstractShape



class MyShape(AbstractShape):
    def __init__(self, area):

        self._area = area
    def area(self) -> np.float32:
        return self._area

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """


        points_count = 3000
        # Generating a set of points I found that evaluate a satisfying result.
        area = np.float32(0)
        shape_cords = contour(points_count)
        for i in range(len(shape_cords) - 1):  # Applying the 'Shoe-Lace' Formula:
            area += np.float32(
                0.5 * (shape_cords[i][0] * shape_cords[i + 1][1] - shape_cords[i + 1][0] * shape_cords[i][1]))

        return np.float32(abs(area))




    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
                Build a function that accurately fits the noisy data points sampled from
                some closed shape.

                Parameters
                ----------
                sample : callable.
                    An iterable which returns a data point that is near the shape contour.
                maxtime : float
                    This function returns after at most maxtime seconds.

                Returns
                -------
                An object extending AbstractShape.
                """


        x_y = []
        maxtime *= 3400

        #Constructing an array of x and y samples from the input noisy contour.
        for i in range(int(maxtime/2)):
            x, y = sample()
            x_y.append([x, y])

        #K_means Alghorithm to get K clusters that will fit as the shape's skeleton (Actually, the 'clean' points)
        clusters = KMeans(n_clusters=32) #The certain K clusters that passed the tests.
        clusters.fit(x_y)

        #Centers of each cluster
        clean_points = []
        for i in clusters.cluster_centers_:
            clean_points.append([i[0], i[1]])
        clean_points.sort()
        begin = clean_points[0]
        sort_centers = [begin]

        # Sort the array using the points' euclidean distance.
        def sort_euclidean(centers_cords, begin):
            distances = []
            for i in range(len(centers_cords)):
                distance = (begin[0] - centers_cords[i][0]) ** 2 + (begin[1] - centers_cords[i][1]) ** 2
                distances.append([distance, [centers_cords[i][0], centers_cords[i][1]]])
            distances.sort()
            lst = []
            for i in distances:
                lst.append(i[1])
            return lst

        #Sort by distances from a random point, using the prior function:
        def recursive_sort(sort_centers, clean_points, begin):
            if len(clean_points) == 2:
                sort_centers.append(clean_points[1])
                sort_centers.append(sort_centers[0])
                return sort_centers
            center = sort_euclidean(clean_points, begin)[1] #We want to get the shape's skeleton
            sort_centers.append(center)                     #One point at a time
            clean_points.remove(begin)
            begin = center
            return recursive_sort(sort_centers, clean_points, begin)
        centers = recursive_sort(sort_centers, clean_points, begin)


        area = 0
        for i in range(len(centers) - 1):  # Applying the 'Shoe-Lace' Formula to evaluate the Polygon's area:
            area += np.float32(
                0.5 * (centers[i][0] * centers[i + 1][1] - centers[i + 1][0] * centers[i][1]))


        return MyShape(abs(area))







##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)



if __name__ == "__main__":
    unittest.main()