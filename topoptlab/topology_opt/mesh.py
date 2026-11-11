# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial 
#
import numpy as np
#
from topoptlab.log_utils import BaseLogger, EmptyLogger
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
from topoptlab.geometry_parser import read_mesh, element_jacobian_det
# MAIN DRIVER
def create_mesh(nelx: int, 
                nely: None | int, 
                nelz: None | int, 
                l: float | list | np.ndarray = 1.,
                mesh_file: None | str = None, 
                logger: None | BaseLogger = None) -> dict:
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
    l : float or list or np.ndarray
        side lengths of elements for regular mesh. For mesh from file, the
        length unit of the mesh (must be float).
    mesh_file : str or None
        file name or path to file containing the mesh readable by gmsh. If
        given, nelx, nely, nelz are ignored.

    Returns
    -------
    mesh_kw : dict
        mesh information. Contains "regular" to indicate regular grid. For
        meshes from file, additionally "coords", "elnodes", "element_type",
        "el_groups", "node_groups" (see geometry_parser.read_mesh).

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
                    "invmapping": _invmapping,
                    "regular": True}
    else:
        #
        if not isinstance(l, float):
            raise ValueError("For irregular meshes, l is the length unit of the mesh. Must be float.")
        #
        mesh_kw = read_mesh(mesh_file)
        mesh_kw["coords"] = mesh_kw["coords"] * l
        ndim = mesh_kw["coords"].shape[1]
        n_el = mesh_kw["elnodes"].shape[0]
        logger.info(f"mesh file: {mesh_file}")
        logger.info(f"number of spatial dimensions: {ndim}")
        logger.info(f"elements: {n_el} {mesh_kw['element_type']}")
        # total volume of mesh
        detJ, w = element_jacobian_det(mesh_kw["coords"][mesh_kw["elnodes"]],
                                       mesh_kw["element_type"])
        mesh_kw.update({"ndim": ndim,
                        "l": l,
                        "n_el": n_el,
                        "n_nodes": mesh_kw["coords"].shape[0],
                        "cellVolume": float((np.abs(detJ)*w).sum()),
                        "mapping": None,
                        "invmapping": None,
                        "regular": False})
    return mesh_kw