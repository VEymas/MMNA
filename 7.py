import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad
import scipy.special

NUM_OF_POINTS = 30
a, b = -10, 10
PRECISION = 10e-6
METHOD = "Gaus"

def F(x):
    return np.exp(x)

def ReplacePoint(start_point, end_point, point):
    return ((start_point + end_point) + (end_point - start_point) * point) / 2

def Mult(x, roots):
    val = 1
    for y in roots:
        val *= (x - y)
    return val

def Test(val, function, start_point, end_point, precision):
    result = quad(function, start_point, end_point)
    diff = np.abs(result[0] - val)
    if diff < precision:
        print("ALL GOOD")
    else:
        print("Can't achieve that precision")
    print(f'dif={diff}')
    print(f'result={result[0]}')
    print(f'val={val}')

def NewtonCotes(start_point, end_point, num_of_points, function, precision):
    points_t = np.linspace(-1, 1, num_of_points + 1, dtype=np.float64)
    f_part = np.array([function(ReplacePoint(start_point, end_point, point)) for point in points_t])
    val = 0
    for i in range(num_of_points + 1):
        roots = points_t.copy()
        roots = np.delete(roots, i)
        p = (Polynomial.fromroots(roots)).integ()
        val += f_part[i] * (p(1) - p(-1)) / Mult(points_t[i], roots)
    val *= (end_point - start_point) / 2
    Test(val, function, start_point, end_point, precision)

def Gaus(start_point, end_point, num_of_points, function, precision):
    points_t = scipy.special.p_roots(num_of_points)[0]
    f_part = np.array([function(ReplacePoint(start_point, end_point, point)) for point in points_t])
    val = 0
    for i in range(num_of_points):
        pol_der = (scipy.special.lpn(num_of_points, points_t[i])[1][-1])**2
        coef = 2 / ((1 - points_t[i]**2) * pol_der)
        val += coef * f_part[i]
    val *= (end_point - start_point) / 2
    Test(val, function, start_point, end_point, precision)

def ClenshawCurtis(start_point, end_point, num_of_points, function, precision):
    maxim = num_of_points // 2
    znam = [1 - 4*k**2 for k in range(maxim)]
    val = 0
    for j in range(num_of_points + 1):
        w = 1 / 2
        mult = 2 * j * np.pi / num_of_points
        ch = 0
        for k in range(1, maxim):
            ch += mult
            w += np.cos(ch) / znam[k]
        w *= 4 / num_of_points
        if not j or j == num_of_points:
            w /= 2
        val += w * function(ReplacePoint(start_point, end_point, np.cos(mult / 2)))
    val *= (end_point - start_point) / 2
    Test(val, function, start_point, end_point, precision)

if METHOD == "NewtonCotes":
    NewtonCotes(a, b, NUM_OF_POINTS, F, PRECISION)
elif METHOD == "Gaus":
    Gaus(a, b, NUM_OF_POINTS, F, PRECISION)
elif METHOD == "ClenshawCurtis":
    ClenshawCurtis(a, b, NUM_OF_POINTS, F, PRECISION)
