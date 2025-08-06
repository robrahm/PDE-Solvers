from fipy import CellVariable, Grid1D, Grid2D, TransientTerm, DiffusionTerm, ConvectionTerm, FaceVariable, Viewer
import numpy as np



"""
This does diffusion on image; this is basic isotropic diffusion
"""
def image_diff(img, alpha, t_end):
    """
    Parameters
        image   : image to be processes
        alpha   : ediffusion coefficient
        t_end   : last time value

    Returns 
        T       : Time values
        U       : Processed Image
    """

    nx, ny = img.shape

    # Make FiPy mesh
    dx = dy = 1.0
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

    # Initialize u with image
    u = CellVariable(mesh=mesh, name="u", value=0.0)
    u.setValue(img.ravel())
    dt = .9 * min(dx**2, dy**2) / (2*np.max(alpha))
    eq = TransientTerm() == DiffusionTerm(coeff = alpha)

    t = 0.0
    while t < t_end: 
        print(t)
        eq.solve(var = u, dt = dt)
        t += dt


    return np.array(u.value.reshape((nx, ny)))


"""
This is a wrapper for fipy FD solver for 2 dim diffusion equation
Solves u_t = alpha u_xx
"""
def solve_heat2d(u0, alpha, dx, dy, t_end, Lx, Ly, dt = None):
    """
    Parameters
        u0      : initial value
        alpha   : diffusion coefficient
        d_      : space and time steps
        t_end   : last time value
        Lx,Ly   : Length of the domain
        dt      : can pass or it is calculed based on CFL

    Returns 
        T       : Time values
        U       : Values of solution at time values
    """
    mesh = Grid2D(dx = dx, nx = int(Lx / dx), dy = dy, ny = int(Ly / dy))

    dt = dt if dt is not None else .9 * min(dx**2, dy**2) / (2*np.max(alpha))
    u = CellVariable(name = "u", mesh = mesh, value = 0.0)
    x = mesh.cellCenters[0].value
    y = mesh.cellCenters[1].value
    u.setValue(u0(x,y))

    u.constrain(0.0, mesh.facesLeft)
    u.constrain(0.0, mesh.facesRight)
    u.constrain(0.0, mesh.facesBottom)
    u.constrain(0.0, mesh.facesTop)

    eq = TransientTerm() == DiffusionTerm(coeff = alpha)

    #viewer = Viewer(u, title = "PDE Solution", figsize = (8,6))

    t = 0.0 
    T = [t]
    while t < t_end: 
        eq.solve(var = u, dt = dt)
        t += dt
        T.append(t)

    #viewer = Viewer(u, title = "PDE Solution", figsize = (8,6))

    return np.array(mesh.cellCenters[0]), np.array(mesh.cellCenters[1]),\
          np.array(u.value.reshape((int(Ly / dy), int(Lx / dx))))





"""
This is a wrapper for fipy FD solver.
Solves u_t = alpha u_xx
"""
def solve_heat(u0, alpha, dx, t_end, L, dt = None, convec = 0, leftval = None, leftdx = None, rightval = None, rightdx = None, 
               g = lambda x, t: 0):
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

    X = np.arange(0, L + dx, dx)
    if not callable(alpha):
        a = lambda x: alpha
    else:
        a = alpha

    if not callable(convec):
        c = lambda x: convec
    else:
        c = convec
    
    mesh = Grid1D(dx = dx, nx = int(L / dx))
    dt = dt if dt is not None else .9 * (dx)**2 / (2*np.max(a(X)))
    #if dt == -1:
    #    dt = .9 * (dx)**2 / (2*np.max(a(X)))
    u = CellVariable(name="u", mesh=mesh, value=0.0)
    x = mesh.cellCenters[0].value
    ar = a(x)
    ac = CellVariable(name="a", mesh=mesh, value = ar)
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
   
    v_values = c(mesh.faceCenters[0])
    v_vector = FaceVariable(mesh=mesh, rank=1, value=(v_values,))

    u[:] = u0(x)
    t = 0.0
    T = [t]
    U = [u.value.copy()]

    while t < t_end:
        
        f = CellVariable(mesh=mesh, value=g(mesh.cellCenters[0], t))
        v_face = FaceVariable(mesh=mesh, value = c(mesh.faceCenters[0]))
        eq = TransientTerm() == DiffusionTerm(coeff=ac) + ConvectionTerm(coeff=v_vector) + f
        eq.solve(var=u, dt=dt)
        t += dt
        T.append(t)
        U.append(u.value.copy())

    return np.array(T), np.array(U)