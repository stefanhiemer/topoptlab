# TOPOLOGY OPTIMIZATION WITH LINEARIZED BUCKLING CRITERIA IN 250 LINES OF MATLAB
# BY F. FERRARI, J.K. GUEST AND O. SIGMUND (SAMO, 2021)
# https://doi.org/10.1007/s00158-021-02854-x
# Python translation by Stefan Hiemer (August 2026), following the style/
# dependencies (numpy, scipy.sparse, scipy.sparse.linalg) of topopt88.py.
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu, eigsh, LinearOperator
from scipy.ndimage import correlate
from scipy.optimize import brentq
from matplotlib import colors
import matplotlib.pyplot as plt


# continuation scheme: increments v by vCn[3] every vCn[2] iterations once
# loop>=vCn[0], as long as v is still below the target vCn[1]
def cnt(v, vCn, loop):
    return v + (loop >= vCn[0])*(v < vCn[1])*((loop % vCn[2]) == 0)*vCn[3]


# relaxed Heaviside projection and its derivatives w.r.t. threshold eta and
# field v
def prj(v, eta, beta):
    v = np.asarray(v).ravel()
    return (np.tanh(beta*eta)+np.tanh(beta*(v-eta))) / \
           (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))


def deta(v, eta, beta):
    v = np.asarray(v).ravel()
    return -beta/np.sinh(beta)*(1/np.cosh(beta*(v-eta)))**2 * \
        np.sinh(v*beta)*np.sinh((1-v)*beta)


def dprj(v, eta, beta):
    v = np.asarray(v).ravel()
    return beta*(1-np.tanh(beta*(v-eta))**2) / \
        (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))


# Kreisselmeier-Steinhauser aggregation function and its derivative
def fKS(p, v):
    v = np.asarray(v)
    return v.max()+np.log(np.sum(np.exp(p*(v-v.max()))))/p


def dKS(p, v, dv):
    v = np.asarray(v)
    dv = np.asarray(dv)
    w = np.exp(p*(v-v.max()))
    return (dv*w[None, :]).sum(axis=1)/w.sum()


def assembly_indices():
    """
    Row/column index pattern of the lower symmetric triangle of an 8x8
    matrix, filled column by column (top to bottom), matching MATLAB's
    ``tril(ones(8))==1`` linear (column-major) indexing order. Used both to
    populate the element stiffness matrix Ke0 and to build the global
    assembly index vectors iK, jK.

    Returns
    -------
    sI, sII : np.ndarray, shape (36,)
        1-based row/column indices into an 8x8 matrix (36 = 8+7+...+1).
    """
    sI, sII = [], []
    for j in range(1, 9):
        sI += list(range(j, 9))
        sII += [j]*(9-j)
    return np.array(sI), np.array(sII)


def ocUpdate(loop, xT, dg0, g1, dg1, ocPar, xOld, xOld1, asy, beta, restartAs):
    """
    MMA-like optimality criteria update with adaptive asymptotes (see
    section 2.3/Appendix of the SAMO paper). Solves the dual problem for the
    single active constraint g1 via bisection/root finding on the KKT
    Lagrange multiplier.

    Parameters
    ----------
    loop : int
        current iteration counter (1-based, as in the MATLAB code).
    xT : np.ndarray, shape (nact,)
        current design variables (active set only).
    dg0 : np.ndarray, shape (nact,)
        sensitivity of the objective w.r.t. xT.
    g1 : float
        value of the (aggregated) constraint.
    dg1 : np.ndarray, shape (nact,)
        sensitivity of the constraint w.r.t. xT.
    ocPar : sequence of 3 floats
        (move limit, gamma for decreasing asymptote, gamma for increasing
        asymptote).
    xOld, xOld1 : np.ndarray, shape (nact,)
        design variables of the previous and second-to-last iteration.
    asy : np.ndarray, shape (nact,2) or None
        asymptotes of the previous iteration.
    beta : float
        current Heaviside projection sharpness (also used to size the
        initial asymptote spacing).
    restartAs : bool
        force re-initialization of the asymptotes (e.g. after a
        continuation step).

    Returns
    -------
    x : np.ndarray, shape (nact,)
        updated design variables.
    asy : np.ndarray, shape (nact,2)
        updated asymptotes.
    lmid : float
        Lagrange multiplier associated with the constraint.
    """
    move = ocPar[0]
    xU = np.minimum(xT+move, 1.0)
    xL = np.maximum(xT-move, 0.0)
    if loop < 2.5 or restartAs:
        asy = np.column_stack((xT-0.5*(xU-xL)/(beta+1),
                                xT+0.5*(xU-xL)/(beta+1)))
    else:
        tmp = (xT-xOld)*(xOld-xOld1)
        gm = np.ones(xT.shape)
        gm[tmp > 0] = ocPar[2]
        gm[tmp < 0] = ocPar[1]
        asy = np.column_stack((xT-gm*(xOld-asy[:, 0]),
                                xT+gm*(asy[:, 1]-xOld)))
    xL = np.maximum(0.9*asy[:, 0]+0.1*xT, xL)
    xU = np.minimum(0.9*asy[:, 1]+0.1*xT, xU)
    # split (+) and (-) parts of the objective/constraint derivatives
    p0_0 = (dg0 > 0)*dg0
    q0_0 = (dg0 < 0)*dg0
    p1_0 = (dg1 > 0)*dg1
    q1_0 = (dg1 < 0)*dg1
    p0 = p0_0*(asy[:, 1]-xT)**2
    q0 = -q0_0*(xT-asy[:, 0])**2
    p1 = p1_0*(asy[:, 1]-xT)**2
    q1 = -q1_0*(xT-asy[:, 0])**2

    def primalProj(lm):
        sp = np.sqrt(p0+lm*p1)
        sq = np.sqrt(q0+lm*q1)
        val = (sp*asy[:, 0]+sq*asy[:, 1])/(sp+sq)
        return np.minimum(xU, np.maximum(xL, val))

    def psiDual(lm):
        xp = primalProj(lm)
        return g1 - ((asy[:, 1]-xT) @ p1_0 - (xT-asy[:, 0]) @ q1_0) + \
            np.sum(p1/np.maximum(asy[:, 1]-xp, 1e-12) +
                   q1/np.maximum(xp-asy[:, 0], 1e-12))

    lmUp = 1e6
    x = xT.copy()
    lmid = -1.0
    if psiDual(0)*psiDual(lmUp) < 0:
        lmid = brentq(psiDual, 0, lmUp)
        x = primalProj(lmid)
    elif psiDual(0) < 0:
        lmid = 0.0
        x = primalProj(lmid)
    elif psiDual(lmUp) > 0:
        lmid = lmUp
        x = primalProj(lmid)
    return x, asy, lmid


def main(nelx, nely, penalK, rmin, ft, ftBC, eta, beta, ocPar, maxit, Lx,
         penalG, nEig, pAgg, prSel, x0=None):
    """
    Topology optimization with linearized buckling criteria, translated
    from "Topology optimization with linearized buckling criteria in 250
    lines of Matlab" (Ferrari, Sigmund, Guest, SAMO 2021).

    Parameters
    ----------
    nelx, nely : int
        number of elements in x/y direction.
    penalK : float
        initial SIMP penalty exponent for the stiffness interpolation.
    rmin : float
        filter radius (element units).
    ft : int
        0/1 plain density filter, 2 filter+projection, 3 filter+projection
        with volume-preserving threshold adaptation.
    ftBC : str
        'N' for a symmetric (Neumann-like) filter boundary, anything else
        for zero padding.
    eta : float
        initial projection threshold.
    beta : float
        initial projection sharpness.
    ocPar : sequence of 3 floats
        (move limit, gamma decrease, gamma increase) for ocUpdate.
    maxit : int
        maximum number of iterations.
    Lx : float
        physical width of the design domain (height follows from the
        nely/nelx aspect ratio).
    penalG : float
        initial SIMP penalty exponent for the stress-stiffness
        interpolation.
    nEig : int
        number of buckling eigenvalues used in the KS aggregation.
    pAgg : float
        initial KS aggregation factor.
    prSel : tuple(str, sequence)
        problem selector, e.g. ('CV',[]) minimize compliance s.t. volume,
        ('VC',[complianceFactor]) minimize volume s.t. compliance,
        ('BCV',[cMax,volfrac]) maximize BLF s.t. compliance and volume,
        ('VCB',[complianceFactor,blfFactor]) minimize volume s.t.
        compliance and BLF.
    x0 : np.ndarray or str, optional
        initial design (element densities, shape (nelx*nely,)) or a path to
        a ``.npy`` file containing it. If omitted a uniform design is used.

    Returns
    -------
    None.

    """
    label, prVal = prSel
    print("Topology optimization with linearized buckling criteria")
    print("ndes: " + str(nelx) + " x " + str(nely) + ", problem: " + label)
    # ------------------------------- PRE 1) MATERIAL AND CONTINUATION PARS
    E0, Emin, nu = 1.0, 1e-6, 0.3
    penalCntK = (25, 1, 25, 0.25)
    penalCntG = (25, 1, 25, 0.25)
    betaCnt = (400, 24, 25, 2)
    pAggCnt = (2e5, 1, 25, 2)
    volfrac = 1.0 if label[0] == 'V' else prVal[-1]
    # --------------------------------------- PRE 2) DISCRETIZATION FEATURES
    Ly = nely/nelx*Lx
    nEl = nelx*nely
    elNrs = np.arange(nEl).reshape(nely, nelx, order='F')
    nodeNrs = np.arange((nely+1)*(nelx+1)).reshape(nely+1, nelx+1, order='F')
    nDof = (nely+1)*(nelx+1)*2
    sI, sII = assembly_indices()
    # ----------------------------------------- elemental stiffness matrix
    c1 = np.array([12, 3, -6, -3, -6, -3, 0, 3, 12, 3, 0, -3, -6, -3, -6, 12,
                   -3, 0, -3, -6, 3, 12, 3, -6, 3, -6, 12, 3, -6, -3, 12, 3,
                   0, 12, -3, 12], dtype=float)
    c2 = np.array([-4, 3, -2, 9, 2, -3, 4, -9, -4, -9, 4, -3, 2, 9, -2, -4,
                   -3, 4, 9, 2, 3, -4, -9, -2, 3, 2, -4, 3, -2, 9, -4, -9, 4,
                   -4, -3, -4], dtype=float)
    Ke = (c1+nu*c2)/(1-nu**2)/24.0                    # lower-tri unique vals
    Ke0 = np.zeros((8, 8))
    Ke0[sI-1, sII-1] = Ke
    Ke0 = Ke0+Ke0.T-np.diag(np.diag(Ke0))              # full elemental matrix
    # base dof (0-based) of the bottom-left node of every element, plus the
    # local dof pattern -> element connectivity matrix cMat (0-based dofs)
    base = 2*nodeNrs[:-1, :-1].reshape(nEl, 1, order='F')+2
    offs = np.array([0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1])
    cMat = base+offs[None, :]
    iK = cMat[:, sI-1].T                                # shape (36,nEl)
    jK = cMat[:, sII-1].T
    Iar = np.sort(np.stack((iK.flatten(order='F'),
                             jK.flatten(order='F')), axis=1), axis=1)[:, ::-1]
    doBuckling = 'B' in label
    if doBuckling:
        # ---------------------------- PERFORM ONLY IF BUCKLING IS ACTIVE ---
        Cmat0 = np.array([[1, nu, 0], [nu, 1, 0],
                           [0, 0, (1-nu)/2]])/(1-nu**2)
        xiG = np.sqrt(1/3)*np.array([-1, 1])
        etaG = xiG.copy()
        wxi = np.array([1.0, 1.0])
        weta = wxi.copy()
        xe = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*(Lx/nelx/2)
        lMat = np.zeros((3, 4))
        lMat[0, 0] = 1
        lMat[1, 3] = 1
        lMat[2, 1:3] = 1

        def dN(xi, zi):
            return 0.25*np.array([[zi-1, 1-zi, 1+zi, -1-zi],
                                   [xi-1, -1-xi, 1+xi, 1-xi]])

        def B0(gradN):
            return lMat @ np.kron(gradN, np.eye(2))

        indM = np.array([1, 3, 5, 7, 16, 18, 20, 27, 29, 34])-1
        t2ind = np.array([2, 3, 4, 6, 7, 9])-1
        iG = iK[indM, :]
        jG = jK[indM, :]
        IkG = np.sort(np.stack((iG.flatten(order='F'),
                                 jG.flatten(order='F')), axis=1),
                       axis=1)[:, ::-1]
        a1 = IkG[:, 1].reshape(nEl, 10)
        a2 = IkG[:, 0].reshape(nEl, 10)
        # x-derivative of the compact (10 entries per element) geometric
        # stiffness matrix representation, built once from unit
        # displacements at the element level
        gradN0 = np.linalg.solve(dN(0, 0) @ xe, dN(0, 0))
        dZdu = np.zeros((10, 8))
        l_idx = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [2, 2], [3, 2],
                           [4, 2], [3, 3], [4, 3], [4, 4]])-1
        for ii in range(8):
            Uvec = np.zeros(8)
            Uvec[ii] = 1.0
            se = Cmat0 @ B0(gradN0) @ Uvec
            S = np.array([[se[0], se[2]], [se[2], se[1]]])
            tt = np.zeros((8, 8))
            for j in range(2):
                for k in range(2):
                    xi, zi = xiG[j], etaG[k]
                    J = dN(xi, zi) @ xe
                    w = wxi[j]*weta[k]*np.linalg.det(J)
                    gradN = np.linalg.solve(J, dN(xi, zi))
                    B1 = np.vstack((np.kron(gradN, np.array([[1, 0]])),
                                     np.kron(gradN, np.array([[0, 1]]))))
                    tt += (B1.T @ np.kron(np.eye(2), S) @ B1)*w
            sel = np.array([1, 3, 5, 7, 19, 21, 23, 37, 39, 55])-1
            dZdu[:, ii] = tt.flatten(order='F')[sel]
        dZdu[t2ind, :] *= 2
    # ---------------------------- PRE 3) LOADS, SUPPORTS, PASSIVE DOMAINS
    fixed = np.arange(2*(nely+1))
    rows_lc = nely//2+np.arange(-8, 9)
    lcDof = 2*nodeNrs[rows_lc, -1]
    modF = 1e-3/Ly/(len(lcDof)-1)
    F = np.zeros(nDof)
    F[lcDof] = -modF
    F[lcDof[0]] /= 2
    F[lcDof[-1]] /= 2
    rows_pasS = (nely//2-1)+np.arange(-9, 11)
    cols_pasS = np.arange(nelx-10, nelx)
    pasS = elNrs[np.ix_(rows_pasS, cols_pasS)].flatten()
    pasV = np.array([], dtype=int)
    free = np.setdiff1d(np.arange(nDof), fixed)
    act = np.setdiff1d(np.arange(nEl), np.union1d(pasS, pasV))
    # ------------------------- PRE 4) FILTER AND PROJECTION OPERATORS
    bc_mode = 'reflect' if ftBC == 'N' else 'constant'
    vals = np.arange(-np.ceil(rmin)+1, np.ceil(rmin))
    dx, dy = np.meshgrid(vals, vals)
    h = np.maximum(0.0, rmin-np.sqrt(dx**2+dy**2))
    Hs = correlate(np.ones((nely, nelx)), h, mode=bc_mode,
                    cval=0.0).flatten(order='F')
    dHs = Hs.copy()

    def fwdfilter(v):
        v2d = np.asarray(v).reshape(nely, nelx, order='F')
        return correlate(v2d, h, mode=bc_mode,
                          cval=0.0).flatten(order='F')/Hs

    def backfilter(v):
        v2d = (np.asarray(v)/dHs).reshape(nely, nelx, order='F')
        return correlate(v2d, h, mode=bc_mode, cval=0.0).flatten(order='F')

    # ----------------------- PRE 5) ALLOCATE AND INITIALIZE OTHER PARAMETERS
    dV = np.zeros(nEl)
    dV[act] = 1.0/nEl
    xpOld, loop, restartAs, ch = 0.0, 0, False, 1.0
    plotL, plotR, muVec = [], [], []
    x = np.zeros(nEl)
    if x0 is not None:
        x[:] = np.load(x0) if isinstance(x0, str) else np.asarray(x0)
    else:
        x[act] = (volfrac*(nEl-len(pasV))-len(pasS))/len(act)
        x[pasS] = 1.0
    xPhys = x.copy()
    phi = np.zeros((nDof, nEig))
    # ------------------------------------------ live plotting setup
    plt.ion()
    if doBuckling:
        fig, axs = plt.subplots(2, 2, figsize=(9, 7))
        axs[0, 0].remove()
        axs[0, 1].remove()
        axDes = fig.add_subplot(2, 1, 1)
        axL, axR, axBLF = axs[1, 0], axs[1, 0].twinx(), axs[1, 1]
    else:
        fig, axDes = plt.subplots()
    im = axDes.imshow(1-xPhys.reshape(nely, nelx, order='F'), cmap='gray',
                       interpolation='none',
                       norm=colors.Normalize(vmin=0, vmax=1))
    axDes.set_title('Current design')
    axDes.tick_params(axis='both', which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
    fig.show()
    # _______________________________________ START OPTIMIZATION LOOP
    xOld = xOld1 = asy = None
    c0 = v0 = None
    g0 = g1 = lmid = 0.0
    muKS0 = cMax = None
    while loop < maxit and ch > 1e-6:
        loop += 1
        # ------------------------------- RL 1) COMPUTE PHYSICAL DENSITIES
        xTilde = fwdfilter(x)
        xPhys[act] = xTilde[act]
        if ft > 1:
            f_ = (prj(xPhys, eta, beta).mean()-volfrac)*(1.0 if ft == 3
                                                           else 0.0)
            while abs(f_) > 1e-6 and label[0] != 'V':
                eta = eta-f_/deta(xPhys, eta, beta).mean()
                f_ = prj(xPhys, eta, beta).mean()-volfrac
            dHs = Hs/dprj(xPhys, eta, beta)
            xPhys = prj(xPhys, eta, beta)
        ch = np.abs(xPhys-xpOld).max()
        xpOld = xPhys.copy()
        # -------------------------- RL 2) SETUP AND SOLVE EQUILIBRIUM EQS
        sK = Emin+xPhys**penalK*(E0-Emin)
        dsK = np.zeros(nEl)
        dsK[act] = penalK*(E0-Emin)*xPhys[act]**(penalK-1)
        sK_full = np.outer(Ke, sK).flatten(order='F')
        K = coo_matrix((sK_full, (Iar[:, 0], Iar[:, 1])),
                        shape=(nDof, nDof)).tocsc()
        K = K+K.T-diags(K.diagonal())
        Kff = K[free, :][:, free].tocsc()
        Kfact = splu(Kff)
        U = np.zeros(nDof)
        U[free] = Kfact.solve(F[free])
        Uc = U[cMat]
        dc = -dsK*np.sum((Uc @ Ke0)*Uc, axis=1)
        if doBuckling:
            # -------------------------- RL 3) STRESS STIFFNESS MATRIX
            sGP = (Cmat0 @ B0(gradN0) @ Uc.T).T
            Z = np.zeros((nEl, 10))
            for j in range(2):
                for k in range(2):
                    xi, zi = xiG[j], etaG[k]
                    J = dN(xi, zi) @ xe
                    w = wxi[j]*weta[k]*np.linalg.det(J)
                    gradN = np.linalg.solve(J, dN(xi, zi))
                    a_, b_ = gradN[0, :], gradN[1, :]
                    B = np.zeros((3, 10))
                    for jj in range(10):
                        p1, p2 = l_idx[jj]
                        B[0, jj] = a_[p1]*a_[p2]
                        B[1, jj] = b_[p1]*b_[p2]
                        B[2, jj] = b_[p2]*a_[p1]+b_[p1]*a_[p2]
                    Z += (sGP @ B)*w
            sG0 = E0*xPhys**penalG
            dsG = np.zeros(nEl)
            dsG[act] = penalG*E0*xPhys[act]**(penalG-1)
            sG = (sG0[:, None]*Z).flatten()
            G1 = coo_matrix((sG, (IkG[:, 0]+1, IkG[:, 1]+1)),
                             shape=(nDof, nDof))
            G2 = coo_matrix((sG, (IkG[:, 0], IkG[:, 1])),
                             shape=(nDof, nDof))
            G = (G1+G2).tocsc()
            G = G+G.T-diags(G.diagonal())
            # ------------------------------ RL 4) BUCKLING EIGENPROBLEM
            Gff = G[free, :][:, free].tocsc()
            Minv = LinearOperator(Kff.shape, matvec=Kfact.solve)
            vals_, vecs_ = eigsh(Gff, k=nEig+4, M=Kff, Minv=Minv, which='SA')
            mu = -vals_
            order_ = np.argsort(mu)[::-1]
            mu = mu[order_]
            eivSort = vecs_[:, order_[:nEig]]
            KEphi = Kff @ eivSort
            normF = np.sqrt(np.sum(eivSort*KEphi, axis=0))
            phi[:] = 0.0
            phi[np.ix_(free, np.arange(nEig))] = eivSort/normF[None, :]
            # ----------------------------- RL 5) SENSITIVITY ANALYSIS OF BLFs
            dkeG = dsG[:, None]*Z
            dkeG[:, t2ind] *= 2
            phiDKphi = np.zeros((nEl, nEig))
            phiDGphi = np.zeros((nEl, nEig))
            adjL = np.zeros((nDof, nEig))
            for j in range(nEig):
                t = phi[:, j]
                tc = t[cMat]
                phiDKphi[:, j] = dsK*np.sum((tc @ Ke0)*tc, axis=1)
                p = t[a1]*t[a2]+t[a1+1]*t[a2+1]
                phiDGphi[:, j] = np.sum(dkeG*p, axis=1)
                tmp = np.zeros(nDof)
                for k in range(8):
                    contrib = (sG0[:, None]*p) @ dZdu[:, k]
                    np.add.at(tmp, cMat[:, k], contrib)
                adjL[:, j] = tmp
            adjV = np.zeros((nDof, nEig))
            adjV[free, :] = Kfact.solve(adjL[free, :])
            adj = np.zeros((nEl, nEig))
            for j in range(nEig):
                vv = adjV[:, j]
                adj[:, j] = dsK*np.sum((Uc @ Ke0)*vv[cMat], axis=1)
            dmu = -(phiDGphi+mu[None, :nEig]*phiDKphi-adj)
        # -------------------------- RL 6) OBJECTIVE AND CONSTRAINTS
        if loop == 1:
            c0 = F @ U
            v0 = xPhys.mean()
        if label == 'CV':
            g0 = (F @ U)/c0
            dg0 = backfilter(dc/c0)
            g1 = xPhys.mean()/volfrac-1
            dg1 = backfilter(dV/volfrac)
        elif label == 'VC':
            g0 = xPhys.mean()/v0
            dg0 = backfilter(dV/v0)
            g1 = (F @ U)/(prVal[-1]*c0)-1
            dg1 = backfilter(dc/(prVal[-1]*c0))
        elif label == 'BCV':
            if loop == 1:
                muKS0 = fKS(pAgg, mu[:nEig])
                g0 = 1.0
                cMax = prVal[0]
            else:
                g0 = fKS(pAgg, mu[:nEig])/muKS0
            dmKS = dKS(pAgg, mu[:nEig], dmu)
            dg0 = backfilter(dmKS/muKS0)
            g1Vec = np.array([F @ U, xPhys.mean()]) / \
                np.array([cMax*c0, volfrac])-1
            dg1c = backfilter(dc/(cMax*c0))
            dg1V = backfilter(dV/volfrac)
            g1 = fKS(pAgg, g1Vec)
            dg1 = dKS(pAgg, g1Vec, np.column_stack((dg1c, dg1V)))
            plotL.append([1/g0/muKS0, 1/mu[0]])
            plotR.append([g1, *g1Vec])
            muVec.append(mu.copy())
        elif label == 'VCB':
            g0 = xPhys.mean()/v0
            dg0 = backfilter(dV/volfrac)
            muKS = fKS(pAgg, mu[:nEig])
            dmKS = dKS(pAgg, mu[:nEig], dmu)
            g1Vec = np.array([prVal[1]*muKS, F @ U]) / \
                np.array([1.0, prVal[0]*c0])-1
            dg1l = backfilter(dmKS*prVal[1])
            dg1c = backfilter(dc/(prVal[0]*c0))
            g1 = fKS(pAgg, g1Vec)
            dg1 = dKS(pAgg, g1Vec, np.column_stack((dg1l, dg1c)))
            plotL.append(g0)
            plotR.append([g1, *g1Vec])
            muVec.append(mu.copy())
        else:
            raise ValueError("unknown problem selector prSel[0]="+label)
        # -------------------------------- RL 7) UPDATE DESIGN VARIABLES
        if loop == 1:
            xOld = x[act].copy()
            xOld1 = xOld.copy()
            asy = None
        xnew, asy, lmid = ocUpdate(loop, x[act], dg0[act], g1, dg1[act],
                                    ocPar, xOld, xOld1, asy, beta, restartAs)
        xOld1 = xOld
        xOld = x[act].copy()
        x[act] = xnew
        # -------------------------------------- RL 8) PRINT AND PLOT RESULTS
        print("It.:{0:2d} g0:{1:7.4f} g1:{2:0.2e} penalK:{3:7.2f} "
              "penalG:{4:7.2f} eta:{5:7.2f} beta:{6:7.1f} ch:{7:0.3e} "
              "lm:{8:0.3e}".format(loop, g0, g1, penalK, penalG, eta, beta,
                                    ch, lmid))
        im.set_array(1-xPhys.reshape(nely, nelx, order='F'))
        if doBuckling:
            axL.cla()
            axR.cla()
            axBLF.cla()
            pL = np.array(plotL)
            pR = np.array(plotR)
            mV = np.array(muVec)
            axL.plot(np.arange(1, loop+1), pL[:, 0] if pL.ndim > 1 else pL,
                      'C0-')
            axR.plot(np.arange(1, loop+1), pR, 'C1--')
            axBLF.plot(np.arange(1, loop+1), 1/mV[:, :min(4, nEig)])
            axL.set_title('Objective and constraint')
            axBLF.set_title('Lowest BLFs')
        fig.canvas.draw()
        plt.pause(0.01)
        # ------------------------------ apply continuation and restart flag
        penalKold, penalGold, betaOld = penalK, penalG, beta
        penalK = cnt(penalK, penalCntK, loop)
        penalG = cnt(penalG, penalCntG, loop)
        beta = cnt(beta, betaCnt, loop)
        pAgg = cnt(pAgg, pAggCnt, loop)
        restartAs = (beta != betaOld) or (penalK != penalKold) or \
            (penalG != penalGold)
    plt.show()
    input("Press any key...")
    return


if __name__ == "__main__":
    # Illustrative default parameters for a cantilever maximizing the first
    # buckling load factor subject to compliance and volume constraints
    # (case 'BCV'). Not necessarily the paper's own example -- adjust freely.
    nelx, nely = 60, 30
    penalK = 3.0
    rmin = 3.0
    ft = 3          # 0/1 density filter, 2 filter+projection, 3 " "+vol.pres.
    ftBC = 'N'
    eta = 0.5
    beta = 2.0
    ocPar = (0.2, 0.65, 1.2)
    maxit = 200
    Lx = 2.0
    penalG = 3.0
    nEig = 6
    pAgg = 20.0
    prSel = ('BCV', [3.5, 0.65])   # cMax=3.5, volfrac=0.65
    import sys
    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        rmin = float(sys.argv[3])
    if len(sys.argv) > 4:
        maxit = int(sys.argv[4])
    main(nelx, nely, penalK, rmin, ft, ftBC, eta, beta, ocPar, maxit, Lx,
         penalG, nEig, pAgg, prSel)
