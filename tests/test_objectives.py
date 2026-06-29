from typing import Callable
from functools import partial 

from numpy import array, zeros, squeeze
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.objectives import vol_frac, am_angle
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

@pytest.mark.parametrize('args, obj_func', 
                         [ ({"xPhys": array([0., 0.5, 1., 1.])[:, None]}, 
                             vol_frac), 
                           ({"nelx": 4, 
                             "nely": 3, 
                             "baseplate": "S",
                             "p": 10.,
                             "ks_exponent": 0.5,
                             "ndim": 2,
                             "xPhys": array([0.]+4*[1.]+2*[0.]+[1.]+4*[0.],order="F")[:,None]}, 
                             am_angle), 
                         ])

def test_objectives_dx(args: dict,
                       obj_func: Callable, 
                       eps: float = 1e-7):
    #
    if ("nelx" in args.keys()) and\
       ("nely" in args.keys()) and\
       ("nelz" not in args.keys()):
        mapping = partial(map_eltoimg, 
                          **args)#nelx=nelx, nely=nely)
        invmapping = partial(map_imgtoel,
                             **args)
    elif ("nelx" in args.keys()) and\
         ("nely" in args.keys()) and\
         ("nelz" in args.keys()):
        mapping = partial(map_eltovoxel,
                          **args)
        invmapping = partial(map_voxeltoel,
                             **args)
    if ("nelx" in args.keys()) and\
       ("nely" in args.keys()): 
       args["mapping"] = mapping 
       args["invmapping"] = invmapping
    #
    obj, rhs_adj, self_adj = obj_func(**args)

    if self_adj is None:
        #  
        xPhys = args["xPhys"]
        #
        rhs_fd = zeros(xPhys.shape[0])
        for j in range(xPhys.shape[0]):
            xPhys_p = xPhys.copy()
            xPhys_m = xPhys.copy()
            xPhys_p[j] += eps
            xPhys_m[j] -= eps
            obj_p = obj_func(**{**args, 
                                "xPhys": xPhys_p})[0]
            obj_m = obj_func(**{**args, 
                                "xPhys": xPhys_m})[0]
            rhs_fd[j] = (obj_p - obj_m) / (2 * eps)
    elif self_adj:
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    assert_almost_equal(squeeze(rhs_adj), 
                        rhs_fd, 
                        decimal=8)

    return 
