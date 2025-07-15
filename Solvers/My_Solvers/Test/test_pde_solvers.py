"""
Unit tests for the solvers.
"""
import numpy as np
from numpy.testing import assert_allclose
from My_Solvers.pde_solvers import fdm
from Pro_Solvers.pro_solvers import solve_heat

alpha = 2
L = 1.0
dx = 0.01
dt = 0.0001
t_end = 0.05
N = 100
C = np.random.rand(N)
def u0(x):
    u = 0
    k = 0
    while k < N:
        u += C[k]*np.cos((k+1) * np.pi * x) / (k+1)
        k += 1
    return u 
tol = 1e-1

def test_fd_1():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [1,0,0], rbc = [1,0,0])
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftval = 0, rightval = 0)
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)

def test_fd():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [1,0,0], rbc = [0,1,0])
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftval = 0, rightdx = 0)
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
