import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

LOWER_BOUND = 0.1
UPPER_BOUND = 2
PRECISION = 5e-1
MAX_F_VALUE = np.exp(3)
STANDART_DEVIATION = 3
METHOD = "MonteCarloMethod"
SAMPLING = "Sobol"

def TargetFunction(x):
    return 5 * x + 7

def ValidateIntegral(lower, upper, func, precision, computed_value):
    actual_value = quad(func, lower, upper)[0]
    error = np.abs(actual_value - computed_value)
    if error <= precision:
        print("GOOD")
    else:
        print("BAD")
    print(f"Computed Integral: {computed_value}, Error: {error}")

def MonteCarloMethod(lower, upper, precision, func, variance):
    interval = upper - lower
    num_samples = int(np.ceil((variance * interval / precision) ** 2 / 12))
    sampled_points = lower + np.random.rand(num_samples) * interval
    integral_estimate = np.mean([func(x) for x in sampled_points]) * interval
    ValidateIntegral(lower, upper, func, precision, integral_estimate)

def GeometricMonteCarlo(start_point, end_point, precision, function, max_val, grid, sigma):
    length = end_point - start_point
    volume = length * max_val
    val = 0
    num_of_points = int(np.ceil( (sigma * length / precision)**2 / 12) )
    print(num_of_points)

    if grid == "Uniform":
        points_x = start_point + np.random.rand(num_of_points) * length
        points_y = np.random.rand(num_of_points) * max_val
    elif grid == "Sobol":
        sampler = qmc.Sobol(d=2, scramble=False)
        points = sampler.random(num_of_points)
        points_x = [x[0] * length + start_point for x in points]
        points_y = [x[1] * max_val for x in points]
    else:
        sampler = qmc.Sobol(d=2, scramble=True)
        points = sampler.random(num_of_points)
        points_x = [x[0] * length + start_point for x in points]
        points_y = [x[1] * max_val for x in points]

    val = 0

    for i in range(num_of_points):
        val += (function(points_x[i]) >= points_y[i])

    val *= volume / num_of_points

    ValidateIntegral(start_point, end_point, function, precision, val)

if METHOD == "MonteCarloMethod":
    MonteCarloMethod(LOWER_BOUND, UPPER_BOUND, PRECISION, TargetFunction, STANDART_DEVIATION)
elif METHOD == "GeometricMonteCarlo":
    GeometricMonteCarlo(LOWER_BOUND, UPPER_BOUND, PRECISION, TargetFunction, MAX_F_VALUE, SAMPLING, STANDART_DEVIATION)
