import numpy as np
from topoptlab.voigt import to_voigt, voigt_pair

def fd_check(matmodel, F, params, eps=1e-8):
    """
    Finite-difference check of constitutive tensor C = dS/dE
    by perturbing F in directions that induce unit Voigt strain perturbations.

    Uses: dE = sym(F^T dF). Choose dF = F^{-T} dE so that dE is realized.
    """
    S0, C_ana = matmodel(F=F, **params)

    ndim = F.shape[-1]
    nvoigt = ndim * (ndim + 1) // 2

    C_fd = np.zeros_like(C_ana)

    # Build Voigt basis strains dE_j (as symmetric matrices)
    for j in range(nvoigt):
        i, k = voigt_pair(j, ndim)  # returns 1-based (i,k)
        i -= 1
        k -= 1

        dE = np.zeros((ndim, ndim), dtype=F.dtype)
        dE[i, k] = 1.0
        dE[k, i] = 1.0  # symmetric; for diagonal this is redundant

        # dF = F^{-T} dE  -> solve(F^T dF = dE)
        Ft = np.swapaxes(F, -1, -2)

        # batch-safe solve: loop batch if needed
        if F.ndim == 2:
            dF = np.linalg.solve(Ft, dE)
            Fp = F + eps * dF
            Fm = F - eps * dF
            Sp, _ = matmodel(F=Fp, **params)
            Sm, _ = matmodel(F=Fm, **params)
            C_fd[:, j] = (Sp - Sm) / (2 * eps)
        else:
            # F shape (...,ndim,ndim)
            batch_shape = F.shape[:-2]
            dF = np.empty_like(F)
            it = np.ndindex(*batch_shape)
            for idx in it:
                dF[idx] = np.linalg.solve(Ft[idx], dE)

            Fp = F + eps * dF
            Fm = F - eps * dF
            Sp, _ = matmodel(F=Fp, **params)
            Sm, _ = matmodel(F=Fm, **params)
            C_fd[..., :, j] = (Sp - Sm) / (2 * eps)

    return C_ana, C_fd

if __name__ == "__main__":
    
    from topoptlab.material_models.stvenant import stvenant_matmodel
    from topoptlab.material_models.neohooke import neohookean_matmodel
    from topoptlab.stiffness_tensors import isotropic_3d
    # deformation
    F = (np.eye(3))[None,:,:]# + np.random.rand(3,3)[None]*0.005
    
    
    # Neo-Hooke
    C_ana, C_fd = fd_check(
        neohookean_matmodel,
        F,
        params=dict(h=np.array([10.0]), mu=np.array([5.0]))
    )
    
    print("analytical: ", C_ana)
    print("finite difference: ", C_fd)
    print("Neo-Hooke error:", np.abs(C_ana - C_fd).max())
    
    # St. Venant
    c_lin = isotropic_3d()
    C_ana, C_fd = fd_check(
        stvenant_matmodel,
        F,
        params=dict(c=c_lin)
    )
    print("analytical: ", C_ana)
    print("finite difference: ", C_fd)
    print("St. Venant error:", np.abs(C_ana - C_fd).max())