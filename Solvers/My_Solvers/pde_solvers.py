from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
import numpy as np
from collections import deque


"""
This is a wrapper for fipy FD solver.
Solves u_t = alpha u_xx
"""
def fdm(u0, alpha, dx, t_end, L, dt = None, lbc = None, rbc = None, g = lambda x, t: 0):
    """
    Parameters
        u0      : initial value
        alpha   : diffusion coefficient
        dx, dt  : space and time steps
        t_end   : last time value
        L       : Length of the domain; the assumption is that we work on [0,L]
        dt      : can pass or it is calculed based on CFL
        ___     : leftval, etc are boundary conditions
        Xbc  : [a,b,c] for au + bu_x = c conditions. Default is Dirichlet

    Returns 
        T       : Time values
        U       : Values of solution at time values
    """
    
    X = np.arange(0, L + dx, dx)
    if not callable(alpha):
        a = lambda x: alpha
    else:
        a = alpha
    U = np.array([u0(X)])
    dt = dt if dt else .9 * (dx)**2 / (2*np.max(a(X)))
    dtdx = dt/(dx**2) if dt else .9 / (2*np.max(a(X)))

    

    t = 0.0 
    T = [t]
    while T[-1] <= t_end:
        T.append(T[-1] + dt)

        al = .5 * (a(X[2:]) + a(X[1:-1]))
        ar = .5 * (a(X[0:-2]) + a(X[1:-1]))
        Ul = U[-1,2:] - U[-1,1:-1]
        Ur = U[-1,1:-1] - U[-1, 0:-2] 
        u = U[-1,1:-1] + dtdx * (al * Ul - ar * Ur)

        #u = U[-1,1:-1] + dtdx * a(X[1:-1]) * (U[-1,0:-2] + U[-1,2:] - 2*U[-1,1:-1]) + dt * g(X[1:-1], T[-1])
        #u += dtdx * (a(X[2:]) - a(X[0:-2])) * (U[-1,2:] - U[-1,0:-2]) / 4
        """
        The next part is boundary conditions.
        """
        if lbc and lbc[1]:
            gp = U[-1,1] - (2 * dx / lbc[1]) * (lbc[2] - lbc[0] * U[-1,0])
            al = .5 * (a(X[0]) + a(X[1]))
            ar = .5 * (a(X[0] + a(-dx)))
            ul = U[-1,1] - U[-1,0]
            ur = U[-1,0] - gp
            u = np.insert(u, 0,U[-1,0] + dtdx * (al * ul - ar * ur))
            #u = np.insert(u, 0, U[-1,0] + dtdx * 2 * (U[-1,1] - 2 * U[-1,0] + gp))
        else:
            u = np.insert(u, 0, lbc[2] / lbc[0] if (lbc and lbc[2] and lbc[0]) else 0)

        
        if rbc and rbc[1]:
            gp = U[-1,-2] + (2 * dx / rbc[1]) * (rbc[2] - rbc[0] * U[-1,-1])
            u = np.append(u, U[-1][-1] + dtdx * alpha * (gp - 2 * U[-1][-1] + U[-1][-2]))

        else: 
            u = np.append(u, rbc[2] / rbc[0] if (rbc and rbc[2] and rbc[0]) else 0)
            
        U = np.vstack((U, u.copy()))


    return np.array(T), np.array(X), np.array(U)
            


    