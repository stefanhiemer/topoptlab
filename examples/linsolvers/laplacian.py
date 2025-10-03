# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from scipy.sparse.linalg import cg, splu, spsolve

from pyamg.classical import ruge_stuben_solver 
from pyamg.aggregation import smoothed_aggregation_solver

from topoptlab.blocksparse_precond import create_primitive_blocks, make_block_preconditioner
from topoptlab.solve_linsystem import laplacian
from topoptlab.linear_solvers import smoothed_jacobi
from topoptlab.multigrid import multigrid_preconditioner, apply_multigrid, vcycle
from topoptlab.amg import create_interpolators_amg

def just_cg(A, b, rtol=1e-5):
    #
    x,info = cg(A,b,rtol=rtol)
    return x

def block_preconditioner(A, b, 
                         nblocks=2, 
                         block_solver=splu,
                         rtol=1e-5):
    
    #
    indices = create_primitive_blocks(A=A, nblocks=nblocks)
    #
    P = make_block_preconditioner(A=A,
                                  block_inds = indices,
                                  solver_func=block_solver)
    #
    x, info = cg(L,b,M=P,rtol=rtol)
    return x

def amg_preconditioner(A, b, 
                       nlevels=2,
                       smoother_fnc=smoothed_jacobi,
                       smoother_kws = {"max_iter": 1,"omega": 0.6},
                       rtol=1e-5):
    
    P = multigrid_preconditioner(A=A, 
                                 b=b, 
                                 x0=np.zeros(A.shape[0]),
                                 create_interpolators=create_interpolators_amg, 
                                 interpolator_kw={"nlevels": nlevels},
                                 smoother_fnc=smoother_fnc,
                                 smoother_kws=smoother_kws)
    x, info = cg(A,b,M=P,rtol=rtol)
    return x

def amg_solver(A, b, 
               nlevels=2,
               smoother_fnc=smoothed_jacobi,
               smoother_kws = {"max_iter": 1,"omega": 0.6},
               tol=1e-5,
               max_cycles=100):
    
    interpolators = create_interpolators_amg(A=L,nlevels=nlevels)
    #
    x = apply_multigrid(x0 = np.zeros(A.shape[0]) ,
                        A = A, b = b, 
                        interpolators= interpolators,
                        cycle = vcycle, 
                        tol=tol,
                        smoother_fnc=smoothed_jacobi,
                        smoother_kws=smoother_kws,
                        max_cycles=max_cycles,
                        nlevels=nlevels)
    return x

def pyamg_rs_preconditioner(A, b, 
                            nlevels=2,
                            rtol=1e-5):
    
    P = ruge_stuben_solver(A=A.tocsr())
                           #strength=('classical', {'theta': 0.25}),
                           #CF=("RS", {'second_pass': False}),
                           #interpolation="direct",
                           #presmoother=('jacobi', {'omega': 0.6}),
                           #postsmoother=('jacobi', {'omega': 0.6}))
    P = P.aspreconditioner(cycle='V') 
    x, info = cg(A,b,M=P,rtol=rtol)
    return x

def pyamg_smagg_preconditioner(A, b, 
                               nlevels=2,
                               smoother_fnc=smoothed_jacobi,
                               smoother_kws = {"max_iter": 1,"omega": 0.6},
                               rtol=1e-5):
    
    P = smoothed_aggregation_solver(A=A.tocsr())
    P = P.aspreconditioner(cycle='V') 
    x, info = cg(A,b,M=P,rtol=rtol)
    return x

if __name__ == "__main__":
    #
    grid_shape=(100,100)
    rtol = 1e-7
    #
    L,b = laplacian( grid_shape )
    # get ideal solution
    x_sol = spsolve(L,b)
    #
    print("block_preconditioner max. residual: ", 
          (x_sol - block_preconditioner(A=L,
                                        b=b,
                                        rtol=rtol)).max())
    #
    #print("AMG preconditioner max. residual: ", 
    #      (x_sol - amg_preconditioner(A=L,
    #                                  b=b,
    #                                  rtol=rtol)).max())
    #
    print("AMG solver max. residual: ", 
          (x_sol - amg_solver(A=L,
                              b=b,
                              tol=rtol)).max())
    #
    print("pyAMG ruge preconditioner max. residual: ", 
          (x_sol - pyamg_rs_preconditioner(A=L,
                                        b=b,
                                        rtol=rtol)).max())
    #
    print("pyAMG smoothed aggreg. preconditioner max. residual: ", 
          (x_sol - pyamg_smagg_preconditioner(A=L,
                                              b=b,
                                              rtol=rtol)).max())