import numpy as np
from sympy import symbols, diff, lambdify

def newton_method(f, start, tol, max_iter=1000):
    x_sym = symbols('x')
    f_prime = diff(f, x_sym)
    f_func = lambdify(x_sym, f)
    f_prime_func = lambdify(x_sym, f_prime)

    x = start
    n = 0

    while n < max_iter:
        f_x = f_func(x)
        f_x_prime = f_prime_func(x)

        if np.abs(f_x_prime) < 1e-12:
            print(f"The derivative is too small in x={x}, the method may not converge.")
            return None

        x_new = x - f_x / f_x_prime

        if np.abs(x_new - x) < tol:
            print(f"he method converged in {n} iterations. The found root: x = {x_new}")
            return x_new

        x = x_new
        n += 1

    print("Newton's method did not converge in the maximum number of iterations.")
    return None

from sympy import sin, cos, exp

x_sym = symbols('x')

test_functions = [
    sin(x_sym),
    cos(x_sym),
    (x_sym - 5) * (x_sym - 10) * (x_sym - 15),
    (x_sym - 1) * (x_sym - 2) * (x_sym - 3) * (x_sym - 4)
]

start_points = [0.5, 1.0, 3.94, 8.62, 16.34, 161234.34, 2.5, 12.32]

for f, start in zip(test_functions, start_points):
    newton_method(f, start, tol=1e-10)