import numpy as np
from numpy.polynomial.polynomial import Polynomial

DEGREE = 10
START_POINT = -1
END_POINT = 1
PRECISION = 1e-6


def Function(x):
    return np.abs(x)


def CreatePoints(start_point, end_point, count_of_points, uniform_grid=True):
    if uniform_grid:
        return np.linspace(start_point, end_point, count_of_points, dtype=np.float64)
    
    chebyshev_nodes = np.polynomial.chebyshev.chebpts2(count_of_points)
    return (end_point - start_point) / 2 * chebyshev_nodes + (end_point + start_point) / 2


def ComputeDividedDifferences(points, function_values):
    n = len(points)
    coef = function_values.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (points[i] - points[i - j])
    return coef


def NewtonInterpolation(start_point, end_point, count_of_points, function, precision, uniform_grid=True):
    points = CreatePoints(start_point, end_point, count_of_points, uniform_grid)
    function_values = np.array([function(x) for x in points])
    divided_differences = ComputeDividedDifferences(points, function_values)
    
    newton_poly = Polynomial([divided_differences[0]])
    basis_poly = Polynomial([1.0])
    
    for i in range(1, count_of_points):
        basis_poly *= Polynomial([-points[i - 1], 1])
        newton_poly += divided_differences[i] * basis_poly
    
    for i in range(count_of_points):
        approx_value = sum(newton_poly.coef[j] * points[i]**j for j in range(len(newton_poly.coef)))
        if np.abs(approx_value - function(points[i])) >= precision:
            print("Can't achieve the required precision")
            print(newton_poly)
            return newton_poly
    
    print("Interpolation successful!")
    print(newton_poly)

NewtonInterpolation(START_POINT, END_POINT, DEGREE + 1, Function, PRECISION, uniform_grid=False)