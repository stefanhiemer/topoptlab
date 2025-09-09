# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from scipy.linalg import lstsq

def f(x):
    """
    Simple demonstration code for the use of the barizilai borwein optimizer
    by minimizing the rosenbrock function in the interval [-1.5,1.5].
    
    Parameters
    ----------
    x : float
        number of variables.
    
    Returns
    -------
    f(x) : float
        function
    """
    return np.sin(x)+np.arctan(x)

def anderson_example(solver="lstsq"):
    """
    Simple demonstration code for the use of the method of moving asymptotes
    minimizing the rosenbrock function in the interval [-1.5,1.5].
    
    Parameters
    ----------
    solver : str
        only "qr" or "lstsq" available.
    
    Returns
    -------
    None
    """
    # initial guess
    x0 = 1
    # Maximum number of iterations.
    k_max = 100
    # Tolerance on the residual.
    tol_res = 1e-6
    # Parameter m.
    m = 3
    # Vector of iterates x.
    x = np.array([[x0], [f(x0)]])
    # Vector of residuals.
    g = f(x) - x
    # Matrix of increments in residuals.
    G_k = np.array([[g[-1,0] - g[-2,0]]])
    # Matrix of increments in x.
    X_k = np.array([[x[-1,0] - x[-2,0]]])
    print(x)
    print(g)
    print(X_k) 
    print(G_k.shape)
    # 
    k = 2
    while k < k_max and np.abs(g[-1]) > tol_res:
        m_k = min(k, m)
        # Solve the least squares problem
        gamma_k = lstsq(G_k,np.zeros(G_k.shape[0]))[0]
        # Compute new iterate and new residual.
        x = np.append(x,x[-1] + g[-1] - (X_k + np.array(G_k))*gamma_k)
        g = np.append(g,f(x[-1]) - x[-1])
        # Update increment matrices with new elements.
        X_k = np.append(X_k,np.array([[x[-1] - x[-2]]]),axis=1)
        G_k = np.append(G_k,np.array([[g[-1] - g[-2]]]),axis=1)
        if X_k.shape[-1] > m_k:
            X_k = X_k[:,-m_k:]
            G_k = G_k[:,-m_k:]
        k = k + 1
        print(G_k.shape)
    # Prints result: Computed fixed point 2.0134445990 after 9 iterations
    print("Computed fixed point {0:.5f} after {1} iterations\n".format(x[-1], k));
    return

if __name__ == "__main__":
    anderson_example()