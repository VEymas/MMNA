import numpy as np

PRECISION = 1e-6
INTERVAL_START, INTERVAL_END = -1, 1

def FunctionToApproximate(x):
    return np.exp(x)

def EvaluatePolynomial(x, coefficients):
    """Evaluates a polynomial at point x using given coefficients."""
    result = 0
    for i in range(coefficients.shape[0]):
        result += coefficients[i] * x**i
    return result

def FindMaxDeviationPoint(coefficients, start, end, function, num_points=1000):
    """Finds the point where the maximum deviation occurs between the function and the polynomial."""
    x_values = np.linspace(start, end, num_points)
    deviations = np.array([np.abs(function(x) - EvaluatePolynomial(x, coefficients)) for x in x_values])
    max_index = np.argmax(deviations)
    return x_values[max_index]

def ComputeRemezApproximation(start, end, degree, function):
    """Computes the best polynomial approximation using the Remez algorithm."""
    points = np.sort(np.random.rand(degree + 2) * (end - start) + start)
    vander_matrix = np.column_stack((np.vander(points, degree + 1), np.array([(-1)**i for i in range(degree + 2)])))
    function_values = np.array([function(x) for x in points])

    error = PRECISION * 100

    while error > PRECISION:
        solution = np.linalg.solve(vander_matrix, function_values)
        polynomial_coefficients = solution[-2::-1]
        delta = solution[-1]

        new_point = FindMaxDeviationPoint(polynomial_coefficients, start, end, function)
        function_residual = function(new_point) - EvaluatePolynomial(new_point, polynomial_coefficients)

        index = np.argwhere(points > new_point)

        if index.shape[0]:
            index = index.min()
            if vander_matrix[index, -1] * delta * function_residual < 0:
                if index == 0:
                    vander_matrix[1:, :] = vander_matrix[:-1, :]
                    function_values[1:] = function_values[:-1]
                    vander_matrix[index, -1] = -vander_matrix[index, -1]
                    points[1:] = points[:-1]
                else:
                    index -= 1
        else:
            index = degree + 1
            if vander_matrix[index, -1] * delta * function_residual < 0:
                vander_matrix[:-1, :] = vander_matrix[1:, :]
                function_values[:-1] = function_values[1:]
                vander_matrix[index, -1] = -vander_matrix[index, -1]
                points[:-1] = points[1:]

        vander_matrix[index, :-1] = np.array([new_point**i for i in range(degree, -1, -1)])
        function_values[index] = function(new_point)
        points[index] = new_point

        error = np.abs(np.abs(function_residual) - np.abs(delta))
    
    return polynomial_coefficients, np.abs(function_residual)

polynomial_degree = 1

polynomial_coeffs, max_deviation = ComputeRemezApproximation(
    start=INTERVAL_START, end=INTERVAL_END, degree=polynomial_degree, function=FunctionToApproximate)

print(f'Maximum deviation = {max_deviation}')
print(f'Polynomial coefficients = {polynomial_coeffs}')