# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin

# nonlinear material model
from topoptlab.material_models.neohooke import neohookean_matmodel

# for elemental stiffness matrix and internal force
from topoptlab.elements.nonlinear_elasticity_2d import _lk_nonlinear_elast_2d

# The boundary condition for cshape
from topoptlab.example_bc.lin_elast import cshape2d

# HuHu regularization
from topoptlab.elements.huhu_2d import _lk_huhu_2d

# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk


# MAIN DRIVER
def Cshape(nelx, nely, nelz=None,
           xPhys=None, penal=3, 
           Emax=0.1e3, Emin=1e-6, nu=0.3,
           lin_solver="cvxopt-cholmod", preconditioner=None,
           assembly_mode="full", l=1.,
           newton_maxit= 100, nsteps=100, rtol=1e-6, 
           file="Cshape",export=True):
    """
    Run a finite element simulation on a regular grid to validate third medium contact (TMC).
    Details of the equations, parameters, please refer to:
    Frederiksen, Andreas Henrik, Ole Sigmund, and Federico Ferrari. 
    "A Matlab code for analysis and topology optimization with Third Medium Contact." 
    arXiv preprint arXiv:2512.00133 (2025).

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction. Only important if ndim is 3.
    xPhys : np.ndarray
        physical SIMP density
    penal : float
        penalty exponent for the SIMP method.
    Emax : float
        (maximum) Young's modulus. If xPhys is None, all elements take this 
        value.
    Emin : float
        minimum Young's modulus for the modified SIMP approach.
    nu : float
        Poissson's ratio.
    solver : str
        solver for linear systems. Check function lin solve for available 
        options.
    preconditioner : str or None
        preconditioner for linear systems. 
    assembly_mode : str
        whether full or only lower triangle of linear system / matrix is 
        created.
    l : float or tuple of length (ndim) or np.ndarray of shape (ndim)
        side lengths of each element.
    newton_maxit : int
        maximum newton iterations.
    newton_maxit : int
        maximum newton iterations.
    nsteps : int
        load steps.
    rtol : float
        FEM solution relative tolerance.
    file : str
        name of output files
    export : bool
        if True, export design as vtk file.

    Returns
    -------
    None.

    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    # 
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    # total number of design variables/elements
    if ndim == 2:
        n = nelx * nely
    elif ndim == 3:
        n = nelx * nely * nelz
    # kv is the element density of void material
    kv=1e-6     
    xPhys = np.ones(n, dtype=float,order='F')*kv 
    
    
    # Define the domain of cshape
    # wall thickness in solid part
    wall_thk   = nely // 5 
    # leave the right region of the cshape as the void  
    c_end = nelx-2
    # upper and lower rectangles
    for j in range(c_end):
        b = j * nely
        xPhys[b:b+wall_thk] = 1.0
        xPhys[b+(nely-wall_thk):b+nely] = 1.0
    # left solids of cshape
    for j in range(wall_thk):
        xPhys[j*nely:(j+1)*nely] = 1.0
    # ndof, edofMat
    nd_ndof = 2
    ndof = (nelx + 1) * (nely + 1) * nd_ndof
    edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx, nely=nely, nnode_dof=nd_ndof)
    # Construct the index pointers for the coo format
    iK, jK = create_matrixinds(edofMat, mode=assembly_mode)

    # Boundary conditions and load vector
    if ndim==2:
        u,f_final,fixed,free,_ = cshape2d(nelx=nelx,nely=nely,nelz=nelz,
                                          ndof=ndof)
    elif ndim==3:
        raise ValueError("Only for 2D validation")
    
    # Nodal coordinates of the structured mesh
    xs = np.linspace(0.0, nelx*l[0], nelx+1)
    ys = np.linspace(nely*l[1], 0.0, nely+1)  
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([xx.ravel(order="F"), yy.ravel(order="F")])
    edof_nodes = (edofMat[:, ::2] // 2).astype(int)  
    xe = coords[edof_nodes]

    # thickness of 2D elements in z direction 
    thickness = 1.0
    # SIMP interpolation of Young's modulus
    Emin = Emax*kv
    E = Emin + (xPhys ** penal) * (Emax - Emin)
    # Neo-Hookean material parameters
    h_el  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_el = E / (2.0 * (1.0 + nu)) 
    # Bulk modulus based on solid stiffness
    kbulk   = Emax / ( 3 * ( 1 - 2 * nu ) )
    # HuHu scaling parameter, following the MATLAB reference implementation
    kr = 1e-6 * l[0] * nelx * ( kbulk + 4/3* (Emax / (2.0 * (1.0 + nu)) ) ) 

    # Incremental load stepping
    for step in range(1, int(nsteps) + 1):
        f = step / float(nsteps) * f_final
        u_step0 = u.copy()  # rollback point if this step fails
        converged = False

        for it in range(int(newton_maxit)):
            ue = u[edofMat].reshape(n, 8)
            # Nonlinear elastic tangent stiffness and internal force
            Ke_e, fint_e = _lk_nonlinear_elast_2d(
                xe=xe,
                ue=ue,
                material_model=neohookean_matmodel,
                material_constants={"h": h_el[:, None], "mu": mu_el[:, None]},
                quadr_method="gauss-lobatto",
                t=thickness,
                nquad=3,
            ) 
            # HuHu tangent stiffness and internal force contribution
            Ke_e_hu, fint_e_hu = _lk_huhu_2d(
                xe=xe,
                ue=ue,
                exponent=5.0,
                kr=kr,
                mode="picard",
                quadr_method="gauss-lobatto",
                t=thickness,
                nquad=3
            ) 
            # Assemble global tangent stiffness matrix
            sK = (Ke_e+Ke_e_hu).reshape(n, -1).T.flatten(order="F")
            K = assemble_matrix(sK=sK, iK=iK, jK=jK, ndof=ndof, solver=lin_solver, springs=None)
            # assemble global internal force
            fint = np.zeros((ndof, 1), dtype=float)
            np.add.at(fint, edofMat.ravel(), (fint_e+fint_e_hu).ravel()[:, None])

            # Residual: rhs = fext - fint
            rhs = f - fint
            rrNorm = float(np.linalg.norm(rhs[free])) / max(1e-16, float(np.linalg.norm(f[free])))
            if rrNorm <= rtol:  
                converged = True
                print(f"[NL step {step}/{nsteps}] converged: it={it}, ||RelRes||={rrNorm:.3e}")
                break
            
            # Newton update: K du = rhs
            K = apply_bc(K=K, solver=lin_solver, free=free, fixed=fixed)
            du_free, fact, precond = solve_lin(K=K, rhs=rhs[free], solver=lin_solver, preconditioner=preconditioner)

            # Backtracking line search
            r0 = float(np.linalg.norm(rhs[free]))          # absolute residual norm on free dofs
            alpha = 1.0
            alpha_min = 1e-8
            max_ls = 20
            u_base = u.copy()
            accepted = False
            for ls in range(max_ls):
                u_trial = u_base.copy()
                u_trial[free, :] += alpha * du_free

                # Re-evaluate the residual at the trial displacement
                ue_trial = u_trial[edofMat].reshape(n, 8)

                _, fint_e_trial = _lk_nonlinear_elast_2d(
                    xe=xe,
                    ue=ue_trial,
                    material_model=neohookean_matmodel, 
                    material_constants={"h": h_el[:, None], "mu": mu_el[:, None]},
                 quadr_method="gauss-lobatto",
                t=thickness,
                nquad=3
                )
                _, fint_e_hu_trial = _lk_huhu_2d(
                    xe=xe,
                    ue=ue_trial,
                    exponent=5.0,
                    kr=kr,
                    mode="picard",
                    quadr_method="gauss-lobatto",
                    t=thickness,
                    nquad=3
                ) 

                fint_trial = np.zeros((ndof, 1), dtype=float)
                np.add.at(fint_trial, edofMat.ravel(), (fint_e_trial+fint_e_hu_trial).ravel()[:, None])

                rhs_trial = f - fint_trial
                r_trial_free = rhs_trial[free]
                r1 = float(np.linalg.norm(r_trial_free))
                # Accept the trial step if the residual does not increase
                if r1 <= r0 * (1.0 + 1e-12):
                    u = u_trial
                    accepted = True
                    break
                alpha *= 0.5
                if alpha < alpha_min:
                    break

            if not accepted:
                u = u_base
                raise RuntimeError(f"Line search failed: r0={r0:.3e}, last r={r1:.3e}, alpha={alpha:.3e}")
            print(f"[NL step {step}/{nsteps}] it={it:02d} ||r||={rrNorm:.3e} ||du||={float(np.linalg.norm(du_free)):.3e} alpha={alpha:.2e}")
            # No line search version:
            # u[free, :] += du_free
            # print(f"[NL step {step}/{nsteps}] it={it:02d} ||r||={rrNorm:.3e} ||du||={float(np.linalg.norm(du_free)):.3e} alpha=1.00e+00")

        if not converged:
            u[:] = u_step0
            raise RuntimeError(f"Newton failed at step {step}/{nsteps}, last ||r||={rrNorm:.3e}")
        if export:
            export_vtk(filename=f"{file}_it{step:04d}", nelx=nelx, nely=nely, nelz=None, xPhys=xPhys, u=u, f=f,elem_size=l)
    return u


if __name__ == "__main__":
    nelx = 82     # 2 elements are for the void on the right as the same in the ref. paper
    nely = 40
    nelz = None
    elem_size=1.0/nelx

    import sys
    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        nelz = int(sys.argv[3])

    Cshape(nelx=nelx, nely=nely, nelz=nelz, l=elem_size)
