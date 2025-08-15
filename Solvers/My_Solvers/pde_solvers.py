from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
from My_Solvers.haar import create_haar_2d
import numpy as np
from collections import deque

"""
This does anisotropic image diffusion but with haar coefficents 
as the "gradient" 
"""
def image_diff_an_haar(img, c = None, K = 1, l = 1, t_end = 10):
    """
    Parameters
        img     : image to be processes; must be 2^N x 2^N in size. 
        c       : the "perona malik" function
        K       : the parameter in the function
        t_end   : last time value

    Returns 
        U       : processed image
    """
    A, H, Coeff, W = create_haar_2d(img)

    c = c if c else lambda x, K: 1 / (1 + x**2/K**2)
    dx = dy = 1.0
    U0 = img
    U = np.zeros(U0.shape)
    U[1:-1, 1:-1] = U0[1:-1, 1:-1]
    t = 0
    dt = 2 * (1/dx**2 + 1/dy**2)
    dt = 1 / dt
    dtdx = dt 
    dtdy = dt 
    num_steps = 0
    while t < Coeff.shape[0]:
        num_steps += 1
        gradU = np.zeros(U.shape)
        gradU[1:-1, 1:-1] = .5 * (U[2:, 1:-1] - U[0:-2, 1:-1])**2 + .5 * (U[1:-1, 2:] - U[1:-1, 0:-2])**2
        D = c(gradU, K)
        C = c(Coeff[t], K)
        print(f"grad is {D[85, 75]} and haar is {C[85,75]} ratio is \
              {D[85,75] / C[85,75]}")
        K = .9 * (1/512**2) * np.sum(np.abs(C))
        dt = .9 / (4 * np.max(C))
        ar = .5 * (C[2:, 1:-1] + C[1:-1, 1:-1])
        al = .5 * (C[0:-2, 1:-1] + C[1:-1, 1:-1])
        au = .5 * (C[1:-1, 2:] + C[1:-1, 1:-1])
        ad = .5 * (C[1:-1, 0:-2] + C[1:-1, 1:-1])
        Ul = U[0:-2, 1:-1] - U[1:-1, 1:-1]
        Ur = U[2:, 1:-1] - U[1:-1, 1:-1]
        Uu = U[1:-1, 2:] - U[1:-1, 1:-1]
        Ud = U[1:-1, 0:-2] - U[1:-1, 1:-1]
        dUx = l * (al * Ul + ar * Ur) + (1 - l) * C[1:-1, 1:-1] * (Ul + Ur)
        dUy = l * (au * Uu + ad * Ud) + (1 - l) *C[1:-1, 1:-1] * (Uu + Ud)
        U[1:-1 , 1:-1] = U[1:-1, 1:-1] +  dtdx * dUx + dtdy * dUy 

        U[0, 1:-1] = U[1, 1:-1]
        U[-1, 1:-1] = U[-2, 1:-1]
        U[1:-1, 0] = U[1:-1, 1]
        U[1:-1, -1] = U[1:-1, -2]
        U[0,0] = .5 * (U[1, 0] + U[0,1])
        U[-1, 0] = .5 * (U[-2,0] + U[-1,1])
        U[0,-1] = .5 * (U[0, -2] + U[1, -1])
        U[-1, -1] = .5 * (U[-1, -2] + U[-2, -1])

        dtdx /= 2
        dtdy /= 2
        t += 1
        
    print(f"num steps ={num_steps}")
    return A, H, Coeff, W, U


    




"""
Basic anisotropic image diffusion 
"""
def image_diff_an(img, c = None, K = 1, l = 1, t_end = 10):
    """
    Parameters
        img     : image to be processes
        c       : the "perona malik" function
        K       : the parameter in the function
        t_end   : last time value

    Returns 
        U       : processed image
    """

    c = c if c else lambda x, K: 1 / (1 + x**2/K**2)
    dx = dy = 1.0
    U0 = img
    U = np.zeros(U0.shape)
    U[1:-1, 1:-1] = U0[1:-1, 1:-1]
    t = 0.0
    dt = 2 * (1/dx**2 + 1/dy**2)
    dt = 1 / dt
    dtdx = dt 
    dtdy = dt 
    num_steps = 0
    while t <= t_end:
        num_steps += 1
        gradU = np.zeros(U.shape)
        gradU[1:-1, 1:-1] = .5 * (U[2:, 1:-1] - U[0:-2, 1:-1])**2 + .5 * (U[1:-1, 2:] - U[1:-1, 0:-2])**2
        C = c(gradU, K)
        K = .9 * (1/512**2) * np.sum(np.abs(C))
        #print(np.sum(np.abs(C)))
        #print((1/512**2) * np.sum(np.abs(C)))
        dt = .9 / (4 * np.max(C))
        ar = .5 * (C[2:, 1:-1] + C[1:-1, 1:-1])
        al = .5 * (C[0:-2, 1:-1] + C[1:-1, 1:-1])
        au = .5 * (C[1:-1, 2:] + C[1:-1, 1:-1])
        ad = .5 * (C[1:-1, 0:-2] + C[1:-1, 1:-1])
        Ul = U[0:-2, 1:-1] - U[1:-1, 1:-1]
        Ur = U[2:, 1:-1] - U[1:-1, 1:-1]
        Uu = U[1:-1, 2:] - U[1:-1, 1:-1]
        Ud = U[1:-1, 0:-2] - U[1:-1, 1:-1]
        dUx = l * (al * Ul + ar * Ur) + (1 - l) * C[1:-1, 1:-1] * (Ul + Ur)
        dUy = l * (au * Uu + ad * Ud) + (1 - l) *C[1:-1, 1:-1] * (Uu + Ud)
        U[1:-1 , 1:-1] = U[1:-1, 1:-1] +  dtdx * dUx + dtdy * dUy 

        U[0, 1:-1] = U[1, 1:-1]
        U[-1, 1:-1] = U[-2, 1:-1]
        U[1:-1, 0] = U[1:-1, 1]
        U[1:-1, -1] = U[1:-1, -2]
        U[0,0] = .5 * (U[1, 0] + U[0,1])
        U[-1, 0] = .5 * (U[-2,0] + U[-1,1])
        U[0,-1] = .5 * (U[0, -2] + U[1, -1])
        U[-1, -1] = .5 * (U[-1, -2] + U[-2, -1])

        t += dt
        
    print(f"num steps ={num_steps}")
    return U


"""
Basic isotropic image diffusion 
"""
def image_diff(img, c = 1, t_end = 10):
    """
    Parameters
        img     : image to be processes
        c       : the "perona malik" function
        t_end   : last time value

    Returns 
        U       : processed image
    """

    dx = dy = 1.0
    U0 = img
    U = np.zeros(U0.shape)
    U[1:-1, 1:-1] = U0[1:-1, 1:-1]
    t = 0.0
    dt = 2 * (1/dx**2 + 1/dy**2)
    dt = 1 / dt
    dtdx = dt 
    dtdy = dt 
    while t <= t_end:
        t += dt
        #print(f"time step ={t}")
        gradU = np.zeros(U.shape)
        gradU[1:-1, 1:-1] = .25 * (U[2:, 1:-1] - U[0:-2, 1:-1])**2 + .25 * (U[1:-1, 2:] - U[1:-1, 0:-2])**2
        C = c(gradU)
        ar = .5 * (C[2:, 1:-1] + C[1:-1, 1:-1])
        al = .5 * (C[0:-2, 1:-1] + C[1:-1, 1:-1])
        au = .5 * (C[1:-1, 2:] + C[1:-1, 1:-1])
        ad = .5 * (C[1:-1, 0:-2] + C[1:-1, 1:-1])
        Ul = U[0:-2, 1:-1] - U[1:-1, 1:-1]
        Ur = U[2:, 1:-1] - U[1:-1, 1:-1]
        Uu = U[1:-1, 2:] - U[1:-1, 1:-1]
        Ud = U[1:-1, 0:-2] - U[1:-1, 1:-1]
        dUx = c * (Ul + Ur)
        dUy = c * (Uu + Ud)
        U[1:-1 , 1:-1] = U[1:-1, 1:-1] +  dtdx * dUx + dtdy * dUy
    
    return U



"""
This is a wrapper for fipy 2D FD solver.
Solves u_t = alpha u_xx
"""
def fdm2D(u0, alpha, dx, dy, t_end, Lx, Ly, dt = None):
    """
    Parameters
        u0      : initial value (a function)
        alpha   : diffusion coefficient
        dx, dt  : space and time steps
        t_end   : last time value
        Lx,Ly   : Length of the domains; the assumption is that we work on [0,Lx]X[0,Ly]
        dt      : can pass or it is calculed based on CFL

    Returns 
        T       : Time values
        U       : Values of solution at time values
    """

    x = np.arange(0, Lx + dx, dx)
    y = np.arange(0, Ly + dy, dy)
    X, Y = np.meshgrid(x, y)
    U0 = u0(X, Y)
    #The following works for Dirichlet conditions
    U = np.zeros(U0.shape)
    U[1:-1, 1:-1] = U0[1:-1, 1:-1]

    t = 0.0
    dt = 2* alpha * (1/dx**2 + 1/dy**2)
    dt = 1 / dt
    dtdx = dt / (dx**2)
    dtdy = dt / (dy**2)
    while t <= t_end:
        t += dt
        dUx = U[2:, 1:-1]  + U[0:-2, 1:-1] - 2*U[1:-1, 1:-1]
        dUy = U[1:-1, 2:]  + U[1:-1, 0:-2] - 2*U[1:-1, 1:-1]
        U[1:-1 , 1:-1] = U[1:-1, 1:-1] + alpha * (dtdx * dUx + dtdy * dUy)
    
    

    return X, Y, U


"""
Solves u_t = alpha u_xx
"""
def fdm(u0, alpha, dx, t_end, L, convec = 0, dt = None, lbc = None, rbc = None, g = lambda x, t: 0):
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

    if not callable(convec):
        c = lambda x: convec
    else:
        c = convec

    

    t = 0.0 
    T = [t]
    while T[-1] <= t_end:
        T.append(T[-1] + dt)
        al = .5 * (a(X[2:]) + a(X[1:-1]))
        ar = .5 * (a(X[0:-2]) + a(X[1:-1]))
        Ul = U[-1,2:] - U[-1,1:-1]
        Ur = U[-1,1:-1] - U[-1, 0:-2] 
        u = U[-1,1:-1] + dtdx * (al * Ul - ar * Ur) + dt * (c(X[2:]) * U[-1,2:] - c(X[0:-2]) * U[-1, 0:-2]) / (2 * dx)
        

        """
        The next part is boundary conditions.
        """
        if lbc and lbc[1]:
            gp = U[-1,1] - (2 * dx / lbc[1]) * (lbc[2] - lbc[0] * U[-1,0])
            al = .5 * (a(X[0]) + a(X[1]))
            ar = .5 * (a(X[0]) + a(-dx))
            ul = U[-1,1] - U[-1,0]
            ur = U[-1,0] - gp
            u = np.insert(u, 0, U[-1,0] + dtdx * (al * ul - ar * ur))
        else:
            u = np.insert(u, 0, lbc[2] / lbc[0] if (lbc and lbc[2] and lbc[0]) else 0)

        
        if rbc and rbc[1]:
            gp = U[-1,-2] + (2 * dx / rbc[1]) * (rbc[2] - rbc[0] * U[-1,-1])
            al = .5 * (a(X[-1]) + a(L + dx))
            ar = .5 * (a(X[-1]) + a(X[-2]))
            ul = gp - U[-1,-1]
            ur = U[-1,-1] - U[-1,-2]
            u = np.append(u, U[-1,-1] + dtdx * (al * ul - ar * ur))
        else: 
            u = np.append(u, rbc[2] / rbc[0] if (rbc and rbc[2] and rbc[0]) else 0)
            
        U = np.vstack((U, u.copy()))


    return np.array(T), np.array(X), np.array(U)
            


    