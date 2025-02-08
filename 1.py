import numpy as np
from scipy.linalg import cholesky, eigh_tridiagonal
from itertools import combinations
from functools import reduce
from operator import mul

MAX_DEGREE = 10
PRECISION = 1e-6

def ComputeIntegral(n):
    return 0 if n % 2 else 2 / (n + 1)

def ScalarProduct(poly1, poly2, shift, integralFunc):
    result = sum(
        coef1 * coef2 * integralFunc(i + j + shift)
        for i, coef1 in enumerate(poly1)
        for j, coef2 in enumerate(poly2)
    )
    return result

def VerifyOrthogonality(matrix, degree, integralFunc):
    for i in range(degree):
        if abs(abs(ScalarProduct(matrix[i, :i+1], matrix[i, :i+1], 0, integralFunc)) - 1) >= PRECISION:
            print("Failed orthogonality check")
            return
        for j in range(i + 1, degree):
            if abs(ScalarProduct(matrix[i, :i+1], matrix[j, :j+1], 0, integralFunc)) >= PRECISION:
                print("Failed orthogonality check")
                return
    print("Orthogonality verified!")
    print(matrix)

def GramSchmidtMethod(degree, integralFunc):
    gramMatrix = np.array([[integralFunc(i + j) for j in range(degree)] for i in range(degree)])
    orthonormalMatrix = np.linalg.inv(cholesky(gramMatrix))
    VerifyOrthogonality(orthonormalMatrix, degree, integralFunc)

def RecurrenceMethod(degree, integralFunc):
    orthoPoly = np.zeros((degree, degree))
    orthoPoly[0, 0] = np.sqrt(0.5)
    betaPrev = 0
    for i in range(1, degree):
        alpha = ScalarProduct(orthoPoly[i - 1], orthoPoly[i - 1], 1, integralFunc)
        orthoPoly[i, 1:i+1] = orthoPoly[i-1, :i]
        orthoPoly[i, :i] -= alpha * orthoPoly[i - 1, :i]
        if betaPrev:
            orthoPoly[i, :i-1] -= betaPrev * orthoPoly[i - 2, :i-1]
        betaPrev = np.sqrt(ScalarProduct(orthoPoly[i], orthoPoly[i], 0, integralFunc))
        orthoPoly[i, :i+1] /= betaPrev
    VerifyOrthogonality(orthoPoly, degree, integralFunc)

def EigenvalueMethod(degree, integralFunc):
    polyMatrix = np.zeros((degree, degree))
    polyMatrix[0, 0] = np.sqrt(0.5)
    alphaValues, betaValues = [], []
    for i in range(1, degree):
        alpha = ScalarProduct(polyMatrix[i - 1], polyMatrix[i - 1], 1, integralFunc)
        alphaValues.append(alpha)
        eigenRoots = eigh_tridiagonal(np.array(alphaValues), np.array(betaValues))[0]
        for j in range(i):
            polyMatrix[i, j] = (-1)**(i - j) * sum(
                reduce(mul, term) for term in combinations(eigenRoots, i - j)
            )
        polyMatrix[i, i] = 1
        normFactor = np.sqrt(ScalarProduct(polyMatrix[i], polyMatrix[i], 0, integralFunc))
        polyMatrix[i, :i+1] /= normFactor
        betaValues.append(ScalarProduct(polyMatrix[i], polyMatrix[i - 1], 1, integralFunc))
    VerifyOrthogonality(polyMatrix, degree, integralFunc)

integrationMethod = ComputeIntegral
methodChoice = "Eigenvalue"

degree = MAX_DEGREE + 1
if methodChoice == 'Gram-Schmidt':
    GramSchmidtMethod(degree, integrationMethod)
elif methodChoice == "Recurrence":
    RecurrenceMethod(degree, integrationMethod)
elif methodChoice == "Eigenvalue":
    EigenvalueMethod(degree, integrationMethod)