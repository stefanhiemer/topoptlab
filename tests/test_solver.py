from numpy.testing import assert_almost_equal

import pytest

from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import mbb_2d,cantilever_2d,cantilever_2d_wrong,cantilever_2d_twoloads,cantilever_2d_twoloads_wrong

@pytest.mark.parametrize('nelx, nely, volfrac, ft, rmin, solver, preconditioner, bcs',
                         [(10,3,0.5,0,2.4,"scipy-direct",None,mbb_2d),
                          (10,3,0.5,1,2.4,"scipy-cg",None,mbb_2d),
                          (10,3,0.5,1,2.4,"scipy-cg","scipy-ilu",mbb_2d),
                          (10,3,0.5,1,2.4,"cvxopt-cholmod",None,mbb_2d),])

def test_compliance_1(nelx, nely, volfrac, 
                      ft, rmin, 
                      solver, preconditioner, 
                      bcs):
    """
    Does exactly the same as function below. Just to allow to have fast and 
    slow tests in same file.
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