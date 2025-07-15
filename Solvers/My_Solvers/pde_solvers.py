from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
import numpy as np



"""
This is a wrapper for fipy FD solver.
Solves u_t = alpha u_xx
"""
def fdm(u0, alpha, dx, t_end, L, dt = None, lbc = None, rbc = None):
    """
    Parameters
        u0      : initial value
        alpha   : diffusion coefficient
        dx, dt  : space and time steps
        t_end   : last time value
        L       : Length of the domain; the assumption is that we work on [0,L]
        dt      : can pass or it is calculed based on CFL
        ___     : leftval, etc are boundary conditions
        robinX  : [a,b,c] for au + bu_x = c conditions. Default is Dirichlet

    Returns 
        T       : Time values
        U       : Values of solution at time values
    """
    
    if lbc is None:
        al = 1 
        bl = 0 
        cl = 0
    else:
        al = lbc[0]
        bl = lbc[1]
        cl = lbc[2]
        
    if rbc is None:
        ar = 1
        br = 0
        cl = 0
    else:
        ar = rbc[0]
        br = rbc[1]
        cr = rbc[2]

    X = np.arange(0, L + dx, dx)
    U = [u0(X)]
    dt = dt if dt else .9 * (dx)**2 / (2*alpha)
    dtdx = dt/(dx**2) if dt else .9 / (2*alpha)

    t = 0.0 
    T = [t]
    while T[-1] <= t_end:
        T.append(T[-1] + dt)
        u = [0]
        k = 1
        for x in X[1:-1]:
            u.append(U[-1][k] + dtdx * alpha * (U[-1][k-1] + U[-1][k+1] - 2*U[-1][k]))
            k += 1

        """
        The next part is boundary conditions.
        """
        if bl == 0:
            u[0] = cl / al
        else:
            gp = U[-1][1] - (2 * dx / bl) * (cl - al * U[-1][0])
            u[0] = U[-1][0] + dtdx * alpha * (U[-1][1] - 2 * U[-1][0] + gp)
        if br == 0:
            u.append(cr / ar)
        else:
            gp = U[-1][-2] + (2 * dx / br) * (cr - ar * U[-1][-1])
            u.append(U[-1][-1] + dtdx * alpha * (gp - 2 * U[-1][-1] + U[-1][-2]))
        U.append(u.copy())

    return np.array(T), np.array(X), np.array(U)
            


    