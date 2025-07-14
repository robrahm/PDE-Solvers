from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
import numpy as np


"""
This is a wrapper for fipy FD solver.
Solves u_t = alpha u_xx
"""
def solve_heat(u0, alpha, dx, dt, t_end, L):
    """
    Parameters
        u0      : initial value
        alpha   : diffusion coefficient
        dx, dt  : space and time steps
        t_end   : last time value
        L       : Length of the domain

    Returns 
        T       : Time values
        U       : Values of solution at time values
    """

    mesh = Grid1D(dx = dx, nx = int(L / dx))
    u = CellVariable(name="u", mesh=mesh, value=0.0)
    x = mesh.cellCenters[0].value
    u[:] = u0(x)

    eq = TransientTerm() == DiffusionTerm(coeff=alpha)
    t = 0.0
    T = [t]
    U = [u.value.copy()]

    while t < t_end:
        #u.constrain(0.0, mesh.facesLeft)
        #u.constrain(0.0, mesh.facesRight)
        eq.solve(var=u, dt=dt)
        t += dt
        T.append(t)
        U.append(u.value.copy())



    return np.array(T), np.array(U)