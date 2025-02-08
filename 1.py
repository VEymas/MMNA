import numpy as np
from scipy.linalg import cholesky, eigh_tridiagonal
from itertools import combinations
from functools import reduce
from operator import mul

MAX_DEGREE = 10
PRECISION = 1e-6
METHOD_CHOICE = "Gram-Schmidt"

def ComputeIntegral(n):
    return 0 if n % 2 else 2 / (n + 1)

def ScalarProduct(poly1, poly2, shift):
    result = sum(
        coef1 * coef2 * ComputeIntegral(i + j + shift)
        for i, coef1 in enumerate(poly1)
        for j, coef2 in enumerate(poly2)
    )
    return result

def VerifyOrthogonality(matrix, degree):
    for i in range(degree):
        if abs(abs(ScalarProduct(matrix[i, :i+1], matrix[i, :i+1], 0)) - 1) >= PRECISION:
            print("Failed orthogonality check")
            return
        for j in range(i + 1, degree):
            if abs(ScalarProduct(matrix[i, :i+1], matrix[j, :j+1], 0)) >= PRECISION:
                print("Failed orthogonality check")
                return
    print("Orthogonality verified!")
    print(matrix)

def GramSchmidtMethod(degree):
    G = np.zeros((degree, degree))
    for i in range(degree):
        for j in range(degree):
            G[i, j] = ComputeIntegral(i + j)
    L = np.linalg.inv(np.linalg.cholesky(G))
    VerifyOrthogonality(L, degree)

def RecurrenceMethod(degree):
    ortho_poly = np.zeros((degree, degree))
    ortho_poly[0, 0] = np.sqrt(0.5)
    beta_prev = 0
    for i in range(1, degree):
        alpha = ScalarProduct(ortho_poly[i - 1], ortho_poly[i - 1], 1)
        ortho_poly[i, 1:i+1] = ortho_poly[i-1, :i]
        ortho_poly[i, :i] -= alpha * ortho_poly[i - 1, :i]
        if beta_prev:
            ortho_poly[i, :i-1] -= beta_prev * ortho_poly[i - 2, :i-1]
        beta_prev = np.sqrt(ScalarProduct(ortho_poly[i], ortho_poly[i], 0))
        ortho_poly[i, :i+1] /= beta_prev
    VerifyOrthogonality(ortho_poly, degree)

def EigenvalueMethod(degree):
    poly_matrix = np.zeros((degree, degree))
    poly_matrix[0, 0] = np.sqrt(0.5)
    alpha_values, beta_values = [], []
    for i in range(1, degree):
        alpha = ScalarProduct(poly_matrix[i - 1], poly_matrix[i - 1], 1)
        alpha_values.append(alpha)
        eigen_roots = eigh_tridiagonal(np.array(alpha_values), np.array(beta_values))[0]
        for j in range(i):
            poly_matrix[i, j] = (-1)**(i - j) * sum(
                reduce(mul, term) for term in combinations(eigen_roots, i - j)
            )
        poly_matrix[i, i] = 1
        norm_factor = np.sqrt(ScalarProduct(poly_matrix[i], poly_matrix[i], 0))
        poly_matrix[i, :i+1] /= norm_factor
        beta_values.append(ScalarProduct(poly_matrix[i], poly_matrix[i - 1], 1))
    VerifyOrthogonality(poly_matrix, degree)

degree = MAX_DEGREE + 1
if METHOD_CHOICE == 'Gram-Schmidt':
    GramSchmidtMethod(degree)
elif METHOD_CHOICE == "Recurrence":
    RecurrenceMethod(degree)
elif METHOD_CHOICE == "Eigenvalue":
    EigenvalueMethod(degree)
