from numpy.testing import assert_almost_equal

import pytest

from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.accelerators import anderson

@pytest.mark.parametrize('obj_ref',
                         [(233.5550913188)])

def test_anderson(obj_ref):
    """
    Test the minimum compliance problem with different filter settings. 
    Does exactly the same as function below. Just to allow to have fast and 
    slow tests in same file.
    """
    #
    # Default input parameters
    nelx = 60
    nely = int(nelx/3)
    volfrac = 0.5
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    x, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, 
                  rmin=rmin, ft=ft, filter_mode="matrix", 
                  optimizer="mma", lin_solver="scipy-direct",
                  nouteriter=1000,file="mbb_2d",
                  accelerator_kw={"accel_freq": 4, 
                                  "accel_start": 20,
                                  "max_history": 4,
                                  "accelerator": anderson,
                                  "damp": 0.9},
                  bcs=mbb_2d,
                  write_log=False,
                  debug=False,display=False,export=False)
    #
    assert_almost_equal(obj,obj_ref,decimal=5)
    return 