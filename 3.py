import numpy as np
from numpy.polynomial.polynomial import Polynomial

DEGREE = 2
START_POINT, END_POINT = -1, 1
PRECISION = 1e-10
METHOD = "lagrange"
NUM_NODES = DEGREE + 1

def AbsoluteFunction(x):
    return np.abs(x)

def GeneratePoints(start, end, num_points, uniform=True):
    if uniform:
        return np.linspace(start, end, num_points, dtype=np.float64)
    return np.cos(np.linspace(0, np.pi, num_points)) * (end - start) / 2 + (end + start) / 2

def VandermondeInterpolation(num_points, start, end, function, precision, uniform=True):
    points = GeneratePoints(start, end, num_points, uniform)
    vander_matrix = np.vander(points, increasing=True)
    coefficients = np.linalg.solve(vander_matrix, function(points))
    
    error = np.max(np.abs(np.polyval(coefficients[::-1], points) - function(points)))
    print("Vandermonde Interpolation:", coefficients if error < precision else "Can't achieve the required precision")

def LagrangeInterpolation(num_points, start, end, function, precision, uniform=True):
    points = GeneratePoints(start, end, num_points, uniform)
    result = Polynomial([0])
    
    for i, x_i in enumerate(points):
        basis = Polynomial.fromroots(np.delete(points, i))
        result += basis * function(x_i) / np.prod(x_i - np.delete(points, i))
    
    error = np.max(np.abs(result(points) - function(points)))
    print("Lagrange Interpolation:", result if error < precision else "Can't achieve the required precision")

def GramIntegral(n):
    return 2 / (n + 1) if n % 2 == 0 else 0

def OrthogonalPolynomialInterpolation(num_points, start, end, function, precision, uniform=True):
    points = GeneratePoints(start, end, num_points, uniform)
    gram_matrix = np.array([[GramIntegral(i + j) for j in range(num_points)] for i in range(num_points)])
    L_inv = np.linalg.inv(np.linalg.cholesky(gram_matrix))
    
    orthogonal_basis = [Polynomial(L_inv[i, :i + 1]) for i in range(num_points)]
    coefficients = np.linalg.solve(
        np.array([[poly(points[i]) for poly in orthogonal_basis] for i in range(num_points)]),
        function(points)
    )
    
    result = sum(coeff * poly for coeff, poly in zip(coefficients, orthogonal_basis))
    error = np.max(np.abs(result(points) - function(points)))
    print("Orthogonal Polynomial Interpolation:", result if error < precision else "Can't achieve the required precision")

if METHOD == "vandermonde":
    VandermondeInterpolation(NUM_NODES, START_POINT, END_POINT, AbsoluteFunction, PRECISION, uniform=False)
elif METHOD == "lagrange":
    LagrangeInterpolation(NUM_NODES, START_POINT, END_POINT, AbsoluteFunction, PRECISION, uniform=True)
elif METHOD == "orthogonal":
    OrthogonalPolynomialInterpolation(NUM_NODES, START_POINT, END_POINT, AbsoluteFunction, PRECISION, uniform=False)