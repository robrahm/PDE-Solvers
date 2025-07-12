"""
Unit tests for the solvers.
"""
import numpy as np
from numpy.testing import assert_allclose
from My_Solvers.ode_solvers import solve

"""
Test Exponential decay first
"""

l = -2
x0 = 1
t0 = 0
t_end = t0 + 2
baseh = .01
f = lambda t, x: l * x
x_exact = lambda t: x0 * np.exp(l*(t - t0))

def test_scalar_exponential_decay_euler():
    t, x = solve(f, t0, x0, t_end, baseh/10, method="euler")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-2)

def test_scalar_exponential_decay_rk4():
    t, x = solve(f, t0, x0, t_end, baseh, method="rk4")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-3)

def test_scalar_exponential_decay_backward_euler():
    t, x = solve(f, t0, x0, t_end, baseh, method="backward_euler")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-2)


"""
Test Exponential growth next
Use x'' = lx with x(t0) = x0
"""
l = 2

def test_scalar_exponential_growth_euler():
    t, x = solve(f, t0, x0, t_end, baseh/10, method="euler")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-2)

def test_scalar_exponential_growth_rk4():
    t, x = solve(f, t0, x0, t_end, baseh, method="rk4")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-3)

def test_scalar_exponential_growth_backward_euler():
    t, x = solve(f, t0, x0, t_end, baseh, method="backward_euler")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-2)

"""
Test stiff

If it truly is stiff, the euler method should fail
"""
l = -1000
st_end = t0 + .5
sx_exact = lambda t: x0 * np.exp(l*(t - t0))

def test_stiff_exponential_decay_euler():
    t, x = solve(lambda t, x: -1000 * x, t0, x0, st_end, baseh, method="euler")
    assert not np.allclose(x[-1], sx_exact(t[-1]), rtol=0, atol=1e-4)

def test_stiff_exponential_decay_backward_euler():
    t, x = solve(lambda t, x: -1000 * x, t0, x0, st_end, baseh, method="backward_euler")
    assert_allclose(x[-1], sx_exact(t[-1]), rtol = 0, atol = 1e-2)


"""
Test non autonomous with curves
"""
x0 = 1
t0 = 0
t_end = t0 + 2
baseh = .01
f = lambda t, x: x*np.sin(t)*np.sin(t)
x_exact = lambda t: np.exp(.5*(t - .5*np.sin(2*t)))

def test_scalar_exponential_na_euler():
    t, x = solve(f, t0, x0, t_end, baseh/10, method="euler")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-2)

def test_scalar_exponential_na_rk4():
    t, x = solve(f, t0, x0, t_end, baseh, method="rk4")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-3)

def test_scalar_exponential_na_backward_euler():
    t, x = solve(f, t0, x0, t_end, baseh, method="backward_euler")
    assert_allclose(x[-1], x_exact(t[-1]), rtol=1e-2)



"""
Test systems
"""
t0 = 0
vx0 = [1,0]
vt_end = t0 + 6
baseh = .01
mf = lambda t, x: np.array([x[1], -x[0]])
vx_exact = lambda t: np.cos(t)
dvx_exact = lambda t: -np.sin(t)

def test_system_exponential_na_euler():
    vt, vx = solve(mf, t0, vx0, vt_end, baseh/10, method="euler")
    assert_allclose(vx[-1,0], vx_exact(vt[-1]), rtol=1e-2)
    assert_allclose(vx[-1,1], dvx_exact(vt[-1]), rtol=1e-2)

def test_system_exponential_na_rk4():
    vt, vx = solve(mf, t0, vx0, vt_end, baseh/10, method="rk4")
    assert_allclose(vx[-1,0], vx_exact(vt[-1]), rtol=1e-2)
    assert_allclose(vx[-1,1], dvx_exact(vt[-1]), rtol=1e-2)

def test_system_exponential_na_backward_euler():
    vt, vx = solve(mf, t0, vx0, vt_end, baseh/10, method="backward_euler")
    assert_allclose(vx[-1,0], vx_exact(vt[-1]), rtol=1e-2)
    assert_allclose(vx[-1,1], dvx_exact(vt[-1]), rtol=1e-2)

