from numpy.testing import assert_almost_equal

import pytest

from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import mbb_2d,cantilever_2d,cantilever_2d_wrong,cantilever_2d_twoloads,cantilever_2d_twoloads_wrong

@pytest.mark.parametrize('nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref',
                         [(60,20,0.5,0,2.4,"matrix",mbb_2d,216.81366),
                          (60,20,0.5,1,2.4,"matrix",mbb_2d,233.71457),
                          (60,20,0.5,0,2.4,"helmholtz",mbb_2d,218.79281),
                          (60,20,0.5,1,2.4,"helmholtz",mbb_2d,237.59964),
                          (60,20,0.5,0,2.4,"matrix",cantilever_2d_twoloads,510.3727841006),
                          (60,20,0.5,0,2.4,"matrix",cantilever_2d,208.4429354360)])

def test_compliance_filters(nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref):
    """
    Test the minimum compliance problem with different filter settings. 
    Does exactly the same as function below. Just to allow to have fast and 
    slow tests in same file.
    """
    #
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, 
                  rmin=rmin, ft=ft, filter_mode=filter_mode,
                  optimizer="oc",
                  bcs=bcs,
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=5)
    return 

@pytest.mark.slow
@pytest.mark.parametrize('nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref',
                         [(160,100,0.4,0,6.0,"matrix",cantilever_2d_wrong,61.4282510690),
                          (150,150,0.4,0,6.0,"matrix",cantilever_2d_twoloads_wrong,69.2037375459),])

def test_compliance_filters_slow(nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref):
    """
    Does exactly the same as function above. Just to allow to have fast and 
    slow tests in same file.
    """
    #
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, rmin=rmin, ft=ft, 
                  filter_mode=filter_mode,optimizer="oc",
                  bcs=bcs,
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=2)
    return 

@pytest.mark.parametrize('optimizer, obj_ref',
                         [("mma",233.4879433637),])

def test_compliance_optimizers(optimizer,obj_ref):
    """
    Test the minimum compliance problem with different filter settings. 
    Does exactly the same as function below. Just to allow to have fast and 
    slow tests in same file.
    """
    #
    nelx, nely, volfrac, ft, rmin, filter_mode = 60,20,0.5,1,2.4,"matrix"
    #
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, 
                  rmin=rmin, ft=ft, filter_mode=filter_mode,
                  optimizer=optimizer,
                  bcs=mbb_2d,
                  display=False,export=False,write_log=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=5)
    return 