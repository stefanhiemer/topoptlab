from numpy.testing import assert_almost_equal

import pytest

from topoptlab.topology_optimization import main
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
                  output_kw = {"file": None,
                               "display": False,
                               "export": False,
                               "write_log": False,
                               "profile": False,
                               "debug": 0})
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
                  output_kw = {"file": None,
                               "display": False,
                               "export": False,
                               "write_log": False,
                               "profile": False,
                               "debug": 0})
    #
    assert_almost_equal(obj,obj_ref,decimal=2)
    return 

@pytest.mark.parametrize('optimizer, obj_ref',
                         [("mma",233.709917),])

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
                  output_kw = {"file": None,
                               "display": False,
                               "export": False,
                               "write_log": False,
                               "profile": False,
                               "debug": 0})
    #
    assert_almost_equal(obj,obj_ref,decimal=5)
    return 

from topoptlab.example_bc.heat_conduction import heatplate_2d
from topoptlab.elements.poisson_2d import lk_poisson_2d

@pytest.mark.parametrize('obj_ref',
                         [(473.7336997084),])

def test_compliance_heat(obj_ref):
    """
    Heatplate 2d test.
    """
    #
    nelx, nely, volfrac, ft, rmin, filter_mode = 40,40,0.4,0,1.2,"matrix"
    #
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, 
                  rmin=rmin, ft=ft, filter_mode=filter_mode,
                  optimizer="oc",
                  bcs=heatplate_2d, lk=lk_poisson_2d,
                  output_kw = {"file": None,
                               "display": False,
                               "export": False,
                               "write_log": False,
                               "profile": False,
                               "debug": 0})
    #
    assert_almost_equal(obj,obj_ref,decimal=3)
    return 