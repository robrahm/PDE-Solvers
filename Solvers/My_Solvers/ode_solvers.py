import numpy as np
from scipy.optimize import root 

def solve(f, t0, x0, t_end, h, method = "rk4"):
    """
    Unified interface for ODE solvers. 

    Parameters: 
        f       : function f(t,x)
        t0      : initial time 
        x0      : initial value (list or np.array)
        t_end   : final time
        h       : step size
        method  : "euler", "rk4" or "backward_euler"

    Returns:
        t_vals, x_vals  : numpay arrays of time and solution values

    """
    if method == "euler":
        return euler(f, t0, x0, t_end, h)
    if method == "rk4":
        return rk4(f, t0, x0, t_end, h)
    if method == "backward_euler":
        return backward_euler(f, t0, x0, t_end, h)
    else:
        raise ValueError(f"Unknown method: {method}")

def euler(f, t0, x0, t_end, h):
    T = [t0]
    X = [np.array(x0, dtype = float)]
    
    while T[-1] < t_end:
        T.append(T[-1] + h)
        X.append(X[-1] + h * f(T[-1],X[-1]))
        
    return np.array(T), np.array(X)


def rk4(f, t0, x0, t_end, h):
    T = [t0]
    X = [np.array(x0, dtype = float)]
    #t, x = t0, np.array(x0, dtype = float)

    while T[-1] < t_end:
        k1 = f(T[-1],X[-1])
        k2 = f(T[-1] + h * .5, X[-1] + h * .5 * k1)
        k3 = f(T[-1] + h * .5, X[-1] + h * .5 * k2)
        k4 = f(T[-1] + h, X[-1] + h * k3)
        T.append(T[-1] + h)
        X.append(X[-1] + (1/6)*h*(k1 + 2 * k2 + 2 * k3 + k4))
        
    return np.array(T), np.array(X)


def backward_euler(f, t0, x0, t_end, h):
    T = [t0]
    X = [np.atleast_1d(x0).astype(float)]

    while T[-1] < t_end:
        T.append(T[-1] + h)
        X.append(alg_solve(lambda x_n: x_n - X[-1] - h * f(T[-1], x_n), X[-1]))

    return np.array(T), np.array(X)

def alg_solve(F, x):
    sol = root(F, x, method='hybr')
    if not sol.success:
        raise RuntimeError(f"Root solve failed at x = {x}: {sol.message}")
    return np.atleast_1d(sol.x)