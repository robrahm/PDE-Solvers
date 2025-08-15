"""
These functions return "haar" data for a given passed in 
vector Y. The assumption is that the length of Y is a 
power of 2 and represents the values of some function on 
the interval [0,1]. The entries of Y are the values of a function 
at the midpoint of the dyadic intervals of length 1/2^N where
N is the length of the vector Y. So, at the smallest scale, they
represent the averave value of the function over the smallest
dyadic inerval. 

These use the well--know identity that <f, h_I> on the children 
is equal to the average of f on the child minus the average of
f on the parent. 
"""

import numpy as np
from math import log2


def create_haar(Y):
    """
    Parameters: 
    Y:      The values of the function on smallest scale (i.e. 
            the average value of f on smallest intervals)

    Returns: 
    A:      Matrix of averages
    H:      <f,h_I>h_I matrix. So, for example, the 0th index row 
            represents the values of <f,h_I>h_I on intervals one 
            size bigger than the smallest interval. Summing over the columns
            of H gives a vector Z that approximates f on the interval. 
    C:      Matrix of (absolute value of) haar coeeficients 
    """

    N = int(log2(Y.size))
    A = np.zeros((N + 1, 2**N))
    A[0,:] = Y 
    for M in range(1, N + 1):
        L = 2**M
        k = 0
        while k*L < 2**N:
            A[M, k*L: (k+1)*L] = .5 * (A[M - 1, k*L] + A[M-1, (k+1)*L - 1])
            k += 1
    H = A[:-1,] - A[1:,]
    m = 1/2**np.arange(N - 1,-1,-1)
    m = m.reshape(N, 1)
    C = np.sqrt(m) * H
    
    return A, H, C

def create_haar_2d(Z):
    """
    Parameters: 
    Y:      The values of the function on smallest scale (i.e. 
            the average value of f on smallest intervals)

    Returns: 
    A:      Matrix of averages
    H:      <f,h_I>h_I matrix. So, for example, the 0th index row 
            represents the values of <f,h_I>h_I on intervals one 
            size bigger than the smallest interval. Summing over the columns
            of H gives a vector Z that approximates f on the interval. 
    C:      Matrix of (absolute value of) haar coeeficients 
    W:      This is sum<f,h_Q>h_Q. So if you graph (X, Y, Z) you should get 
            the graph of the function on which Z is based.
    """
    N = int(log2(Z.shape[0]))
    A = np.zeros((N + 1, 2**N, 2**N))
    A[0] = Z

    t = 1
    while t <= N:
        scale = 2**t
        r = 0
        while r * scale <= N:
            c = 0
            while c * scale <= N:
                tl = A[t - 1, r * scale, c * scale]
                tr = A[t - 1, r * scale, (c + 1) * scale - 1]
                bl = A[t - 1, (r + 1) * scale - 1, c * scale]
                br = A[t - 1, (r + 1) * scale - 1, (c + 1) * scale - 1]
                A[t, r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = .25 * (tl + tr + bl + br)

                c += 1
            r += 1
        t += 1
    H = A[:-1] - A[1:]
    c = np.sqrt(1/2**np.linspace(N - 1, 0, N, endpoint = True).reshape(N, 1, 1))
    C = c * H

    W = A[-1].copy()
    t = 0
    while t < N:
        W += H[t]
        t += 1
    return A, H, C, W

    
