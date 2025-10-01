from numpy.testing import assert_almost_equal

import pytest

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import cantilever_2d,cantilever_2d_wrong
from topoptlab.geometries import sphere

@pytest.mark.parametrize('nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref',
                         [(60,20,0.5,0,2.4,"matrix",cantilever_2d,221.9720563724)])

def test_compliance_1(nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref):
    """
    Does exactly the same as function below. Just to allow to have fast and
    slow tests in same file.
    """
    #
    center = (nelx/3 - 1, nely/2 - 1)
    radius = nely/3
    pass_el = sphere(nelx=nelx, nely=nely, center=center, radius=radius)
    #
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, rmin=rmin, ft=ft,
                  el_flags=pass_el,filter_mode=filter_mode,optimizer="oc",
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

@pytest.mark.slow
@pytest.mark.parametrize('nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref',
                         [(150,100,0.5,0,5.0,"matrix",cantilever_2d_wrong,53.9494361626),])

def test_compliance_2(nelx, nely, volfrac, ft, rmin, filter_mode, bcs, obj_ref):
    """
    Does exactly the same as function below. Just to allow to have fast and
    slow tests in same file.
    """
    #
    center = (nelx/3 - 1, nely/2 - 1)
    radius = nely/3
    pass_el = sphere(nelx=nelx, nely=nely, center=center, radius=radius)
    #
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, rmin=rmin, ft=ft,
                  el_flags=pass_el,filter_mode=filter_mode,optimizer="oc",
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
