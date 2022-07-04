import autograd.numpy as np
from autograd import grad

def newton(fun, gradient, initial_guess, epsilon, iterations):
    xn = initial_guess
    if isinstance(fun(xn), complex):
        print('Complex, try a positive initial guess.')
    else:
        for i in range(0, iterations):
            fxn = fun(xn)
            if abs(fxn) < epsilon:
                print('Solution found after', i, 'iterations.')
                return round(xn, 4)
            dfxn = gradient(float(xn))
            if dfxn == 0:
                print('Zero derivative. No solution Found.')
            xn = xn - fun(xn)/gradient(float(xn))
        print('Exceed Max Iterations. No Solution Found.')

def fun(x):
    y = np.sin(x)+x**2-20
    return y
gradient = grad(fun)
print(newton(fun, gradient, 100, 1e-12, 100))
