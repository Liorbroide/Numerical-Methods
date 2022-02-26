"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
from assignment2 import Assignment2
import numpy as np
import time
import random
from assignment1 import Assignment1


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        def trapezoid(f, a, b, n):
            n -= 1
            x = np.linspace(a, b, n+1)
            y = []
            for i in x:
                y.append(f(i))
            right_limit = y[1:]
            left_limit = y[:-1]
            dev = (b - a)/n
            res = np.float32((dev/2) * np.sum(right_limit + left_limit))
            return res
        if n < 3:
            return trapezoid(f, a, b, n)
        n -= 1
        if n % 2 == 1:
            n = n - 1
        dx = (b - a) / n
        xs = np.linspace(a, b, n + 1)
        y = [f(x) for x in xs]
        res = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
        return np.float32(res)


    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        intersection = Assignment2() #Obtaining all of the intersecting points of f1 and f2
        int_pts = intersection.intersections(f1, f2, 1, 100, maxerr=0.001)
        inter_lst = []
        for i in int_pts:
            inter_lst.append(i)
        if len(inter_lst) < 2:
            return None
        y = lambda x: f1(x) - f2(x)
        area = np.float32(0)
        for i in range(len(inter_lst) - 1): #Integrating between each set of two intersection points.
            area += np.float32(abs(self.integrate(y, inter_lst[i], inter_lst[i+1], 200)))
        return np.float32(area)



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        f = RESTRICT_INVOCATIONS(10)(f1)
        r = ass3.integrate(f, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
