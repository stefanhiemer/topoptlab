import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def homogenization(lx, ly, lambda_, mu, phi, x,
                   debug=False):
    # Deduce discretization
    nely, nelx = x.shape
    x = x.flatten()
    # Stiffness matrix consists of two parts, one belonging to lambda and one belonging to mu. Same goes for load vector
    dx = lx / nelx
    dy = ly / nely
    nel = nelx * nely
    keLambda, keMu, feLambda, feMu = elementMatVec(dx / 2, dy / 2, phi)
    if debug:
        print('--- keLambda ---')
        print(keLambda)
        print('--- Mu ---')
        print(keLambda)
        print('--- feLambda ---')
        print(feLambda)
        print('--- feMu ---')
        print(feMu)
    # Node numbers and element degrees of freedom for full (not periodic) mesh
    # unique dofs:
    ndof = 2*nelx*nely
    # build indices for periodic matrix
    edofMat = np.zeros((nelx*nely,8),int)
    elx,ely = np.arange(nelx)[:,None], np.arange(nely-1)[None,:]
    n1 = (nely*elx+ely).flatten()
    n2 = (nely*(elx+1)+ely).flatten()
    yperiodic = np.arange(nely-1,nely*nelx+1,nely)
    edofMat[np.setdiff1d(np.arange(nelx*nely), yperiodic)] = np.column_stack((
                                                2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                                                2*n2, 2*n2+1, 2*n1, 2*n1+1))
    # periodicity in x direction
    edofMat[-nely:] = edofMat[-nely:]%ndof
    # periodic in y direction
    edofMat[yperiodic,:4] = ((np.arange(nelx)[:,None] + np.array([[0,0,1,1]]))\
                             * 2 * nely + np.array([[0,1,0,1]]))%ndof
    edofMat[yperiodic,4:6] = edofMat[yperiodic-1,2:4]
    edofMat[yperiodic,6:] = edofMat[yperiodic-1,:2]
    if debug:
        print('--- edofMat ---')
        print(edofMat)
    # ASSEMBLE STIFFNESS MATRIX
    # Indexing vectors
    iK = np.tile(edofMat,8).flatten()
    jK = np.repeat(edofMat,8).flatten()
    # Material properties in the different elements
    lambda_ = lambda_[0] * (x == 0) + lambda_[1] * (x == 1)
    mu = mu[0] * (x == 0) + mu[1] * (x == 1)
    if debug:
        print("--- lambda ---")
        print(lambda_)
        print("--- mu ---")
        print(mu)
    # The corresponding stiffness matrix entries
    sK = (keLambda.flatten()[:,None]*lambda_[None,:] + \
         keMu.flatten()[:,None]*mu[None,:]).flatten('F')
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    if debug:
        print('--- iK ---')
        print(iK)
        print('--- jK ---')
        print(jK)
        print('--- sK ---')
        print(sK)
        print('--- K ---')
        print(K)
    # LOAD VECTORS AND SOLUTION
    # Assembly three load cases corresponding to the three strain cases
    iF = np.tile(edofMat.T, (3,1)).flatten('F')
    jF = np.concatenate([np.zeros((8,nel,), dtype=int),
                         np.ones((8,nel,), dtype=int),
                         np.full((8,nel,),2, dtype=int)]).flatten('F')
    sF = (feLambda.flatten('F')[:,None]*lambda_[None,:] + \
          feMu.flatten('F')[:,None]*mu[None,:]).flatten('F')
    F = coo_matrix((sF, (iF, jF)), shape=(ndof, 3)).tocsc()
    if debug:
        print('--- iF ---')
        print(iF)
        print('--- jF ---')
        print(jF)
        print('--- sF ---')
        print(sF)
        print('--- F ---')
        print(F)
    # Solve (remember to constrain one node)
    chi = np.zeros((ndof, 3))
    chi[2:, :] = spsolve(K[2:, 2:], F[2:, :].toarray())
    if debug:
        print('--- chi ---')
        print(chi)
    # HOMOGENIZATION
    # The displacement vectors corresponding to the unit strain cases
    chi0 = np.zeros((nel, 8, 3))
    # The element displacements for the three unit strains
    chi0_e = np.zeros((8, 3))
    ke = keMu + keLambda  # Here the exact ratio does not matter, because
    fe = feMu + feLambda  # it is reflected in the load vector
    chi0_e[[2, 4, 5, 6, 7], :] = np.linalg.solve(ke[np.ix_([2, 4, 5, 6, 7],
                                                           [2, 4, 5, 6, 7])],
                                                 fe[[2, 4, 5, 6, 7]])
    if debug:
        print('--- ke ---')
        print(ke)
        print('--- fe ---')
        print(fe)
        print('--- chi0_e ---')
        print(chi0_e)
    # epsilon0_11 = (1, 0, 0)
    chi0[:, :, 0] = np.kron(chi0_e[:, 0].T, np.ones((nel, 1)))
    # epsilon0_22 = (0, 1, 0)
    chi0[:, :, 1] = np.kron(chi0_e[:, 1].T, np.ones((nel, 1)))
    # epsilon0_12 = (0, 0, 1)
    chi0[:, :, 2] = np.kron(chi0_e[:, 2].T, np.ones((nel, 1)))
    CH = np.zeros((3, 3))
    cellVolume = lx * ly
    for i in range(3):
        for j in range(3):
            sumLambda = ((chi0[:, :, i] - chi[edofMat, i]) @ keLambda) * \
                         (chi0[:, :, j] - chi[edofMat,j])
            sumMu = ((chi0[:, :, i] - chi[edofMat,i]) @ keMu) * \
                     (chi0[:, :, j] - chi[edofMat,j])
            sumLambda = np.sum(sumLambda, axis=1)#.reshape((nely, nelx))
            sumMu = np.sum(sumMu, axis=1)#.reshape((nely, nelx))
            # Homogenized elasticity tensor
            CH[i, j] = np.sum(lambda_ * sumLambda + mu * sumMu)
    CH = CH / cellVolume
    print('--- Homogenized elasticity tensor ---')
    print(CH)
    return CH

def elementMatVec(a, b, phi):
    # Constitutive matrix contributions
    CMu = np.diag([2, 2, 1])
    CLambda = np.zeros((3, 3))
    CLambda[0:2, 0:2] = 1

    # Two Gauss points in both directions
    xx = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    yy = xx
    ww = [1, 1]

    # Initialize
    keLambda = np.zeros((8, 8))
    keMu = np.zeros((8, 8))
    feLambda = np.zeros((8, 3))
    feMu = np.zeros((8, 3))
    L = np.zeros((3, 4))
    L[0, 0] = 1
    L[1, 3] = 1
    L[2, 1:3] = 1

    for ii in range(len(xx)):
        for jj in range(len(yy)):
            # Integration point
            x = xx[ii]
            y = yy[jj]

            # Differentiated shape functions
            dNx = 1 / 4 * np.array([-(1 - y), (1 - y), (1 + y), -(1 + y)])
            dNy = 1 / 4 * np.array([-(1 - x), -(1 + x), (1 + x), (1 - x)])

            # Jacobian
            J = np.dot(
                np.array([dNx, dNy]),
                np.array([
                    [-a, a, a + 2 * b / np.tan(phi * np.pi / 180), 2 * b / np.tan(phi * np.pi / 180) - a],
                    [-b, -b, b, b]
                ]).T
            )

            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            invJ = 1 / detJ * np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])

            # Weight factor at this point
            weight = ww[ii] * ww[jj] * detJ

            # Strain-displacement matrix
            G = np.block([[invJ, np.zeros((2, 2))], [np.zeros((2, 2)), invJ]])
            dN = np.zeros((4, 8))
            dN[0, 0:8:2] = dNx
            dN[1, 0:8:2] = dNy
            dN[2, 1:8:2] = dNx
            dN[3, 1:8:2] = dNy
            B = np.dot(np.dot(L, G), dN)

            # Element matrices
            keLambda += weight * (np.dot(np.dot(B.T, CLambda), B))
            keMu += weight * (np.dot(np.dot(B.T, CMu), B))

            # Element loads
            feLambda += weight * (np.dot(np.dot(B.T, CLambda),
                                         np.diag([1, 1, 1])))
            feMu += weight * (np.dot(np.dot(B.T, CMu),
                                     np.diag([1, 1, 1])))

    return keLambda, keMu, feLambda, feMu


def check_elementMatVec():

    keLambda, keMu, feLambda, feMu = elementMatVec(1/200, 1/200, 90)
    print(keLambda)
    print(keMu)
    print(feLambda)
    print(feMu)
    print(keLambda + keMu)

    return

if __name__ == "__main__":
    #
    #check_elementMatVec()
    # Example usage:
    #lambda = nu*E / ( (1+nu)*(1-2*nu) ) # 2D
    #mu = E/(2*(1+nu))
    Es = [1e-3,1e0]
    nus = [0.3,0.3]
    lambda_ = []
    mu = []
    for i in range(len(Es)):
        E,nu = Es[i],nus[i]
        lambda_.append(nu*E / ( (1+nu)*(1-2*nu) ))
        mu.append(E/(2*(1+nu)))
    lambda_ = np.array(lambda_)
    mu = np.array(mu)
    lambda_ = 2*mu*lambda_ / (lambda_+2*mu)
    #lambda_ = [0.01, 2.]
    #mu = [0.02, 4.]
    lx = ly = 1.0
    phi = 90
    x = np.eye(2,dtype=int)
    homogenization(lx=lx, ly=ly, lambda_=lambda_, mu=mu, phi=phi, x=x,
                   debug = False)
    import sys
    sys.exit()
    lx = ly = 1
    phi = 90
    np.random.seed(0)
    x = np.random.randint(0,2,(20,20))
    homogenization(lx, ly, lambda_, mu, phi, x)
