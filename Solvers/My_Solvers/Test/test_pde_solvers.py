"""
Unit tests for the solvers.
"""
import numpy as np
from numpy.testing import assert_allclose
from My_Solvers.pde_solvers import fdm
from Pro_Solvers.pro_solvers import solve_heat
from scipy.interpolate import interp1d

alpha = 2
L = 1.0
dx = 0.01
dt = 0.0001
t_end = 0.001
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
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 
    


#This is dirichlet again, but tests for the case when no BCs are passed which defaults to DD BCs
def test_fd_1_5():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L)
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftval = 0, rightval = 0)
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 

def test_fd_2():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [1,0,0], rbc = [0,1,0])
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftval = 0, rightdx = 0)
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 

def test_fd_3():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [0,1,0], rbc = [0,1,0])
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftdx = 0, rightdx = 0)
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 

def test_fd_4():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [0,1,0], rbc = [1,0,0])
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftdx = 0, rightval = 0)
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 





def test_fd_1_nh():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [1,0,0], rbc = [1,0,0], g = lambda x,t: (t**2)*np.cos(x))
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftval = 0, rightval = 0, g = lambda x,t: (t**2)*np.cos(x))
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 

def test_fd_2_nh():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [1,0,0], rbc = [0,1,0], g = lambda x,t: (t**2)*np.cos(x))
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftval = 0, rightdx = 0, g = lambda x,t: (t**2)*np.cos(x))
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 

def test_fd_3_nh():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [0,1,0], rbc = [0,1,0], g = lambda x,t: (t**2)*np.cos(x))
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftdx = 0, rightdx = 0, g = lambda x,t: (t**2)*np.cos(x))
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 

def test_fd_4_nh():
    T1, X1, U1 = fdm(u0, alpha, dx, t_end, L , lbc = [0,1,0], rbc = [1,0,0], g = lambda x,t: (t**2)*np.cos(x))
    T, U = solve_heat(u0, alpha, dx, t_end, L, leftdx = 0, rightval = 0, g = lambda x,t: (t**2)*np.cos(x))
    assert(np.abs(U1[-1,-1] - U[-1,-1]) <  tol)
    assert(np.abs(U1[-1,0] - U[-1,0]) < tol)
    X = np.arange(dx/2, L, dx)
    inU = interp1d(X, U[-1], kind='cubic', fill_value = "extrapolate")
    U = inU(X1)
    diff = dx*np.sum((U1[-1] - U)**2)
    assert(diff < 1e5 * tol) 
