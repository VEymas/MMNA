import numpy as np
from numpy.polynomial.polynomial import Polynomial
import math

DEGREE = 30
START_POINT = -1
END_POINT = 1
PRECISION = 1e-2

def FunctionValue(x, step):
    if step % 4 == 0:
        return np.sin(x)
    elif step % 4 == 1:
        return np.cos(x)
    elif step % 4 == 2:
        return -np.sin(x)
    else:
        return -np.cos(x)

def GenerateInterpolationPoints(start, end, num_points, uniform=True):
    half_n = (num_points + 1) // 2

    if uniform:
        points = np.linspace(start, end, half_n, dtype=np.float64)
    else:
        chebyshev_nodes = np.polynomial.chebyshev.chebroots([0] * half_n + [1])
        chebyshev_nodes[0], chebyshev_nodes[-1] = start, end
        points = chebyshev_nodes

    extra_points = np.random.choice(half_n, num_points - half_n)
    points = np.sort(np.append(points, points[extra_points]))
    
    return points

def HermiteNewtonInterpolation(start, end, num_points, function, precision, uniform=True):
    points = GenerateInterpolationPoints(start, end, num_points, uniform)
    divided_differences = [[function(x, 0) for x in points]]

    for i in range(1, num_points):
        current_differences = []
        for j in range(num_points - i):
            if np.abs(points[j] - points[j + i]) < 1e-10:
                current_differences.append(function(points[j], i) / math.factorial(i))
            else:
                numerator = divided_differences[i - 1][j] - divided_differences[i - 1][j + 1]
                denominator = points[j] - points[j + i]
                current_differences.append(numerator / denominator)
        divided_differences.append(current_differences)

    result_polynomial = Polynomial([divided_differences[0][0]])

    for i in range(1, num_points):
        result_polynomial += divided_differences[i][0] * Polynomial.fromroots(points[:i])

    return VerifyInterpolation(points, result_polynomial, function, precision)

def VerifyInterpolation(points, polynomial, function, precision):
    unique_points, counts = np.unique(points, return_counts=True)
    is_accurate = True

    for i in range(len(unique_points)):
        for k in range(counts[i]):        
            derivative_at_x = polynomial.deriv(k)(unique_points[i])
            actual_value = function(unique_points[i], k)
            if np.abs(derivative_at_x - actual_value) >= precision:
                is_accurate = False

    if is_accurate:
        print("Interpolation is successful.")
    else:
        print("Approximation error is too high.")

    return polynomial


np.random.seed(42)
hermite_polynomial = HermiteNewtonInterpolation(START_POINT, END_POINT, DEGREE + 1, FunctionValue, PRECISION, True)
print(hermite_polynomial)
