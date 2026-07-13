# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Tuple, Union
from functools import partial 
#
import numpy as np
#
from topoptlab.log_utils import BaseLogger, EmptyLogger
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
# MAIN DRIVER
def create_mesh(nelx: int, 
                nely: Union[None,int], 
                nelz: Union[None,int], 
                l: Union[float,List,np.ndarray] = 1.,
                mesh_file: Union[None,str] = None, 
                logger: Union[None,BaseLogger] = None) -> Dict:
    """
    Create mesh from file or standard cuboid mesh.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction. If None, simulation is 2d.
    mesh_file : str or None
        file name or path to file containing the mesh readable by gmsh.

    Returns
    -------
    None.

    """
    #
    logger = logger or EmptyLogger()
    #
    if mesh_file is None:
        #
        if nely is None:
            ndim = 1
            raise NotImplementedError("1D currently not implemented.")
        elif nelz is None:
            ndim = 2
        else:
            ndim = 3
        # check if log file exists and if True delete
        logger.info(f"number of spatial dimensions: {ndim}")
        logger.info("elements: "+" x ".join([f"{nelx}",f"{nely}",f"{nelz}"][:ndim]))
        # total number of elements
        n_el = int(np.prod([nelx, nely, nelz][:ndim]))
        #
        if isinstance(l, float):
            l = np.array([l for i in np.arange(ndim)])
        # bundle all mesh-related information; irregular-mesh support will extend this
        if ndim == 2:
            _mapping = partial(map_eltoimg,  
                               nelx=nelx, 
                               nely=nely)
            _invmapping = partial(map_imgtoel,  
                                  nelx=nelx, 
                                  nely=nely)
        else:
            _mapping = partial(map_eltovoxel, 
                               nelx=nelx, 
                               nely=nely, 
                               nelz=nelz)
            _invmapping = partial(map_voxeltoel, 
                                  nelx=nelx, 
                                  nely=nely, 
                                  nelz=nelz)
        mesh_kw = {"nelx": nelx,
                    "nely": nely,
                    "nelz": nelz,
                    "ndim": ndim,
                    "l": l,
                    "n_el": n_el,
                    "cellVolume": float(np.prod(l) * n_el),
                    "mapping": _mapping,
                    "invmapping": _invmapping}
    else:
        raise NotImplementedError("Irregular meshes not yet implemented.")
    return mesh_kw