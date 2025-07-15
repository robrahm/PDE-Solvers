from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
import numpy as np


"""
This is a wrapper for fipy FD solver.
Solves u_t = alpha u_xx
"""
def solve_heat(u0, alpha, dx, t_end, L, dt = -1, leftval = None, leftdx = None, rightval = None, rightdx = None):
    """
    Parameters
        u0      : initial value
        alpha   : diffusion coefficient
        dx, dt  : space and time steps
        t_end   : last time value
        L       : Length of the domain
        dt      : can pass or it is calculed based on CFL
        ___     : leftval, etc are boundary conditions

    Returns 
        T       : Time values
        U       : Values of solution at time values
    """
    
    mesh = Grid1D(dx = dx, nx = int(L / dx))
    if dt == -1:
        dt = .9 * (dx)**2 / (2*alpha)
    u = CellVariable(name="u", mesh=mesh, value=0.0)
    x = mesh.cellCenters[0].value
    """
    If left and or right vals are given set those; if fluxes are give set those;
    as you can see, flux has precedence. 
    """
    
    if leftdx is not None:
        u.faceGrad.constrain(leftdx, mesh.facesLeft)
    elif leftval is not None:
        u.constrain(leftval, mesh.facesLeft)   

    if rightdx is not None:
        u.faceGrad.constrain(rightdx, mesh.facesRight)
    elif rightval is not None:
        u.constrain(rightval, mesh.facesRight) 
   
    eq = TransientTerm() == DiffusionTerm(coeff=alpha)

    u[:] = u0(x)

    
    t = 0.0
    T = [t]
    U = [u.value.copy()]

    while t < t_end:
        
        eq.solve(var=u, dt=dt)
        t += dt
        T.append(t)
        U.append(u.value.copy())

    return np.array(T), np.array(U)