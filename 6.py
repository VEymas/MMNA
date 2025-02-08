import numpy as np

def F(x):
    return np.abs(x)

def RunThroughMethod(a, b, c, d):
    n = len(b)
    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)
    x = np.zeros(n)

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denominator = b[i] + a[i - 1] * alpha[i - 1]
        if np.abs(denominator) < 1e-10:
            raise ValueError("Zero division in run through method!")
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i - 1] * beta[i - 1]) / denominator

    x[-1] = (d[-1] - a[-1] * beta[-1]) / (b[-1] + a[-1] * alpha[-1])
    
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    
    return x

def ComputeDelta(f, x1, x2, x3, h1, h2):
    f1 = (f(x1) - f(x2)) / h1
    f2 = (f(x2) - f(x3)) / h2
    return 6 * (f1 - f2)

def GenerateRightPart(points, segments, function):
    return np.array([
        ComputeDelta(function, points[i + 2], points[i + 1], points[i], segments[i + 1], segments[i])
        for i in range(len(segments) - 1)
    ])

def CubicSplineInterpolation(start, end, func, numPoints, uniform=True):
    points = np.linspace(start, end, numPoints) if uniform else np.sort(np.random.rand(numPoints) * (end - start) + start)
    points[0], points[-1] = start, end
    segments = np.diff(points)

    dVector = GenerateRightPart(points, segments, func)
    mainDiag = 2 * (segments[:-1] + segments[1:])
    sideDiag = segments[1:-1]
    coefficients = RunThroughMethod(sideDiag, mainDiag, sideDiag, dVector)
    coefficients = np.insert(np.append(coefficients, 0), 0, 0)  # Natural spline conditions

    splines = []
    for i in range(1, numPoints):
        f_i = func(points[i - 1])
        df = (func(points[i]) - f_i) / segments[i - 1]
        diff = (coefficients[i] - coefficients[i - 1]) / (2 * segments[i - 1])
        
        a3 = (coefficients[i] - coefficients[i - 1]) / (6 * segments[i - 1])
        a2 = coefficients[i - 1] / 2 - diff * points[i - 1]
        a1 = df - points[i - 1] * coefficients[i - 1] + diff * points[i - 1] ** 2
        a0 = f_i - points[i - 1] * df + coefficients[i - 1] * points[i - 1] ** 2 / 2 - diff * points[i - 1] ** 3 / 3
        splines.append([a3, a2, a1, a0])
    
    return np.array(splines), points

def EvaluateSpline(x, points, coeffs):
    for i in range(len(points) - 1):
        if points[i] <= x <= points[i + 1]:
            a3, a2, a1, a0 = coeffs[i]
            dx = x - points[i]
            return a3 * dx**3 + a2 * dx**2 + a1 * dx + a0
    return None

np.random.seed(42)
a, b, n = -1, 1, 9
coeffs, points = CubicSplineInterpolation(a, b, F, n, uniform=True)
print(coeffs)
