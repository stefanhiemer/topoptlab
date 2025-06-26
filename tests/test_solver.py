from numpy.testing import assert_almost_equal

import pytest

from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import mbb_2d,mbb_3d

@pytest.mark.parametrize('nelx, nely, volfrac, ft, rmin, solver, preconditioner, bcs',
                         [(10,3,0.5,0,2.4,"scipy-direct",None,mbb_2d),
                          (10,3,0.5,1,2.4,"scipy-cg",None,mbb_2d),
                          (10,3,0.5,1,2.4,"scipy-cg","scipy-ilu",mbb_2d),
                          (10,3,0.5,1,2.4,"cvxopt-cholmod",None,mbb_2d)])

def test_compliance_1(nelx, nely, volfrac,
                      ft, rmin,
                      solver, preconditioner,
                      bcs):
    """
    Checks that different solvers yield same end result and no accumulation of
    floating errors causes serious deviations with small MBB problem. Take
    standard scipy solver as comparison.
    """
    #
    x, obj_ref = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0,
                      rmin=rmin, ft=ft,
                      filter_mode="matrix",optimizer="oc",
                      bcs=bcs,lin_solver="scipy-direct",
                      display=False,export=False,write_log=False)
    #
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0,
                  rmin=rmin, ft=ft,
                  filter_mode="matrix",optimizer="oc",
                  bcs=bcs,lin_solver=solver,preconditioner=preconditioner,
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=5)
    return

@pytest.mark.parametrize('nelx, nely, nelz, volfrac, ft, rmin, solver, preconditioner, assembly_mode, bcs',
                         [(10,3,None,0.5,1,2.4,"cvxopt-cholmod",None,"full",mbb_2d),
                          (10,3,None,0.5,1,2.4,"cvxopt-cholmod",None,"lower",mbb_2d),
                          (10,3,3,0.5,1,2.4,"cvxopt-cholmod",None,"full",mbb_3d),
                          (10,3,3,0.5,1,2.4,"cvxopt-cholmod",None,"lower",mbb_3d),])

def test_assembly(nelx, nely, nelz, volfrac,
                  ft, rmin,
                  solver, preconditioner,assembly_mode,
                  bcs):
    """
    Checks that different assembly modes yield same end result and no
    accumulation of floating errors causes serious deviations.
    """
    #
    x, obj_ref = main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=3.0,
                      rmin=rmin, ft=ft,
                      filter_mode="matrix",optimizer="oc",
                      bcs=bcs,lin_solver="scipy-direct",
                      display=False,export=False,write_log=False)
    #
    x, obj = main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=3.0,
                  rmin=rmin, ft=ft,assembly_mode=assembly_mode,
                  filter_mode="matrix",optimizer="oc",
                  bcs=bcs,lin_solver=solver,preconditioner=preconditioner,
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=5)
    return
