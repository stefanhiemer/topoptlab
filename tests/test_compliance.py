from numpy.testing import assert_almost_equal

import pytest

from topoptlab.compliance_minimization import main 

@pytest.mark.parametrize('ft',
                         [(0),(1)])

def test_mbb_density_filter(ft):
    
    #
    # Default input parameters
    nelx = 60  # 180
    nely = int(nelx/3)  # 60
    volfrac = 0.5  # 0.4
    rmin = 0.04*nelx
    x, obj = main(nelx, nely, volfrac, penal=3.0, rmin=rmin, ft=ft, 
                  passive=False,pde=False,solver="oc",
                  display=False,export=False,write_log=False)
    #
    if ft == 0:
        assert_almost_equal(obj,216.81,decimal=2)
    if ft == 1:
        assert_almost_equal(obj,233.71,decimal=2) 
    return 