# SPDX-License-Identifier: GPL-3.0-or-later
from warnings import warn 

import numpy as np

import gmsh

from topoptlab.fem import get_integrpoints
from topoptlab.elements.linear_interval import shape_functions_dxi as dxi_interval
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi as dxi_quad
from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi as dxi_hex

def mesh_cadfile(cad_file : str,
                 output_file : str, 
                 etype : str, 
                 show_gui: bool = False) -> None:
    """
    Mesh a given CAD file (usually step format, but whatever gmsh can read
    should work). 

    Parameters
    ----------
    cad_file : str
        path to the cad file.
    output_file : str
        path to mesh file.
    etype : str
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    #
    gmsh.initialize()
    #
    gmsh.open(cad_file)
    #
    if etype in ["tri","quad"]: 
        dim = 2 
    else: 
        dim = 3
    #
    if etype == "quad":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
    if etype == "hex":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
    #
    gmsh.model.mesh.generate(dim)
    gmsh.write(output_file)
    # launch GUI
    if show_gui:
        gmsh.fltk.run()
    #
    gmsh.finalize()
    return

def read_mesh(meshfile: str,
              ndim: None | int = None) -> dict:
    """
    Read a Gmsh mesh file and return node coordinates, element connectivity
    and physical groups. The mesh must contain exactly one element type in
    its highest topological dimension. Supported are linear intervals (bars),
    bilinear quadrilaterals and trilinear hexahedra. Nodes not used by any
    element are removed and nodes renumbered. If the topological dimension
    equals the spatial dimension, elements with negative jacobian are flipped.

    Parameters
    ----------
    meshfile : str
        Path to the mesh file readable by Gmsh.
    ndim : int or None
        spatial dimension. If None, inferred from the last coordinate axis
        with non-zero extent.

    Returns
    -------
    mesh : dict
        "coords" : np.ndarray, shape (n_nodes,ndim)
            node coordinates.
        "elnodes" : np.ndarray, shape (n_el,n_nodes_el)
            node indices of each element in topoptlab node ordering.
        "element_type" : str
            "interval", "quadrilateral" or "hexahedron".
        "el_groups" : dict[str, np.ndarray]
            element indices of physical groups with same topological
            dimension as elements.
        "node_groups" : dict[str, np.ndarray]
            node indices of physical groups with lower topological
            dimension (e. g. boundaries).
        If a group has no name, ``group_<tag>`` is used.
    """
    # start gmsh
    gmsh.initialize()
    # read file
    gmsh.open(meshfile)
    # topological dimension of mesh
    eldim = gmsh.model.getDimension()
    # get nodes and node coordinates
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords).reshape(-1, 3)
    # get element information
    eltypes, eltags, elnodetags = gmsh.model.mesh.getElements(eldim)
    # check if multiple element types used. if yes, raise error
    if len(eltypes) != 1:
        gmsh.finalize()
        raise RuntimeError("mesh must contain exactly one element type in the highest dimension. If physical groups are defined, gmsh only saves elements in physical groups (unless Mesh.SaveAll=1).")
    # gmsh element type -> topoptlab element type. Gmsh node ordering of
    # these coincides with the topoptlab ordering.
    _eltypes = {1: "interval", 3: "quadrilateral", 5: "hexahedron"}
    if eltypes[0] not in _eltypes:
        name = gmsh.model.mesh.getElementProperties(eltypes[0])[0]
        gmsh.finalize()
        raise ValueError(f"Element type {name} not supported.")
    element_type = _eltypes[eltypes[0]]
    n_nodes_el = gmsh.model.mesh.getElementProperties(eltypes[0])[3]
    # map node tags to node indices
    perm = np.argsort(nodeTags)
    sorted_nodetags = nodeTags[perm]
    elnodes = perm[np.searchsorted(sorted_nodetags,
                                   elnodetags[0])].reshape(-1, n_nodes_el)
    # map global element tags -> element index
    eltags = np.asarray(eltags[0])
    perm_el = np.argsort(eltags)
    sorted_eltags = eltags[perm_el]
    # loop over physical groups
    el_groups, node_groups = {}, {}
    for dim_pg, tag_pg in gmsh.model.getPhysicalGroups():
        #
        name = gmsh.model.getPhysicalName(dim_pg, tag_pg) or f"group_{tag_pg}"
        # lower dimensional groups are stored as node sets
        if dim_pg < eldim:
            tags = gmsh.model.mesh.getNodesForPhysicalGroup(dim_pg, tag_pg)[0]
            node_groups[name] = perm[np.searchsorted(sorted_nodetags, tags)]
            continue
        # loop entities of phys. group
        idx = []
        for ent_tag in gmsh.model.getEntitiesForPhysicalGroup(dim_pg, tag_pg):
            eltypes_ent, eltags_ent, _ = gmsh.model.mesh.getElements(dim_pg, ent_tag)
            # empty entity
            if len(eltypes_ent) == 0:
                continue
            tags_ent = np.asarray(eltags_ent[0])
            pos = np.searchsorted(sorted_eltags, tags_ent)
            idx.extend(perm_el[pos])
        el_groups[name] = np.asarray(idx, dtype=int)
    #
    gmsh.finalize()
    # remove nodes not attached to elements and renumber
    used, elnodes = np.unique(elnodes, return_inverse=True)
    elnodes = elnodes.reshape(-1, n_nodes_el)
    coords = coords[used]
    new = np.full(nodeTags.shape[0], -1)
    new[used] = np.arange(used.shape[0])
    node_groups = {name: new[nodes][new[nodes] >= 0]
                   for name, nodes in node_groups.items()}
    # infer spatial dimension
    if ndim is None:
        extent = np.ptp(coords, axis=0) > 0
        ndim = max(int(np.flatnonzero(extent).max()) + 1, eldim)
    coords = coords[:, :ndim]
    # flip elements with negative orientation
    if eldim == ndim and eldim > 1:
        detJ, _ = element_jacobian_det(coords[elnodes], element_type)
        flip = detJ.mean(axis=1) < 0
        if eldim == 2:
            elnodes[flip] = elnodes[flip][:, [0, 3, 2, 1]]
        else:
            elnodes[flip] = elnodes[flip][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return {"coords": coords,
            "elnodes": elnodes,
            "element_type": element_type,
            "el_groups": el_groups,
            "node_groups": node_groups}

def element_jacobian_det(xe: np.ndarray,
                         element_type: str,
                         nquad: int = 2) -> np.ndarray:
    """
    Determinant of the jacobian of the isoparametric mapping at the
    Gauss-Legendre quadrature points. If topological and spatial dimension
    differ (e. g. bars in 2D/3D), sqrt(det(J^T J)) is returned which is
    always positive.

    Parameters
    ----------
    xe : np.ndarray, shape (n_el,n_nodes_el,ndim)
        coordinates of element nodes.
    element_type : str
        "interval", "quadrilateral" or "hexahedron".
    nquad : int
        number of quadrature points per direction.

    Returns
    -------
    detJ : np.ndarray, shape (n_el,n_quad)
        determinant at each quadrature point.
    w : np.ndarray, shape (n_quad)
        quadrature weights.
    """
    #
    if element_type == "interval":
        shape_functions_dxi, eldim = dxi_interval, 1
    elif element_type == "quadrilateral":
        shape_functions_dxi, eldim = dxi_quad, 2
    elif element_type == "hexahedron":
        shape_functions_dxi, eldim = dxi_hex, 3
    else:
        raise ValueError(f"Unknown element type {element_type}.")
    #
    x, w = get_integrpoints(ndim=eldim, nq=nquad, method="gauss-legendre")
    dN = shape_functions_dxi(*x.T).reshape(x.shape[0], xe.shape[1], eldim)
    # jacobian shape (n_el,n_quad,ndim,eldim)
    J = np.einsum("eni,qnj->eqij", xe, dN)
    if J.shape[-2] == eldim:
        return np.linalg.det(J), w
    else:
        return np.sqrt(np.linalg.det(np.swapaxes(J, -1, -2) @ J)), w

def cad_to_mesh(file: str, 
                mesh_dim: int = 3, 
                mesh_file: str = "output.msh",
                transfinite_transform: bool = True,
                npoints: int = 10, 
                check_rect: bool = True,
                check_hex: bool = True,
                show_gui: bool = False) -> None:
    """
    Take a CAD file (so far only STEP tested) and mesh it with GMSH with 
    quadrilateral or hexahedral elements. The mesh is then written to the 
    mesh file.

    Parameters
    ----------
    file : str
        file name.
    mesh_dim : int 
        dimension of mesh
    mesh_file : str 
        name of mesh file
    show_gui : bool
        show final mesh with GMSH Gui

    Returns
    -------
    None 
    
    """
    warn("At the current stage, this can only parse transfinite geometries (regular mesh).")
    # initialize model
    gmsh.initialize()
    gmsh.model.add(file.rsplit(".",1)[0])
    # load the STEP file into OpenCASCADE kernel (apparently latter is 
    # necessary)
    gmsh.model.occ.importShapes(file)
    gmsh.model.occ.synchronize()
    # transfinite transform
    if transfinite_transform:
        for dim, tag in gmsh.model.getEntities(dim=1):
            #
            gmsh.model.mesh.setTransfiniteCurve(tag, 
                                                npoints)  
        # 
        surfaces = gmsh.model.getEntities(dim=2)
        for s in surfaces:
            # transfinite transformation
            try:
                if transfinite_transform: 
                    gmsh.model.mesh.setTransfiniteSurface(s[1])
            except Exception as e:
                print(f"Could not apply transfinite to surface {s[1]}: {e}")
            # create rectangular surfaces instead of triangular
            gmsh.model.mesh.setRecombine(2, s[1])
        # define physical groups for each volume
        volumes = gmsh.model.getEntities(dim=3)
        for v in volumes:
            # transfinite transformation
            try:
                if transfinite_transform: 
                    gmsh.model.mesh.setTransfiniteVolume(v[1])
            except Exception as e:
                print(f"Could not apply transfinite to volume {v[1]}: {e}")
            # create hexahedral volumes instead of tetrahedral
            gmsh.model.mesh.setRecombine(3, v[1])
    # create physical groups
    surfaces = gmsh.model.getEntities(dim=2)
    for s in surfaces:
        # define physical groups for each surface
        gmsh.model.addPhysicalGroup(s[0], [s[1]])
        gmsh.model.setPhysicalName(s[0], s[1], f"Surface_{s[1]}")
    # define physical groups for each volume
    volumes = gmsh.model.getEntities(dim=3)
    for v in volumes:
        # define physical groups for each surface
        gmsh.model.addPhysicalGroup(v[0], [v[1]])
        gmsh.model.setPhysicalName(v[0], v[1], f"Volume_{v[1]}")
    # check that all elements are rectangular/hexahedral
    admissible = []
    if check_rect:
        admissible.append(3)
    if check_hex:
        admissible.append(5)
    for entity in gmsh.model.getEntities():
        # Dimension and tag of the entity:     
        dim = entity[0]     
        tag = entity[1]
        if dim < 2:
            continue
        print("dim, tag", dim, tag)
        #
        ent_type = gmsh.model.getType(dim, tag)
        ent_name = gmsh.model.getEntityName(dim, tag)
        # mesh nodes for the entity (dim, tag)
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)
        # Get the mesh elements for the entity (dim, tag):     
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        #
        if any([el_type not in admissible for el_type in elemTypes]):
            gmsh.finalize()
            raise ValueError("Inadmissible elements used. At the moment only quadrilateral and hexahedral are supported, not tetraeder. Make sure to generate meshes, that allow for this.")
            
        print("elemTypes", 
               elemTypes)
        print()
    #elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
    #for etype in element_types:
    #    name, dim, order, num_nodes = gmsh.model.mesh.getElementProperties(etype)
    #    print(f"Element Type: {name}, Dimension: {dim}, Order: {order}, Nodes: {num_nodes}")
    # write out mesh file
    gmsh.write(mesh_file)
    # GUI
    if show_gui:
        gmsh.fltk.run()
    # 
    gmsh.finalize()
    return