# SPDX-License-Identifier: GPL-3.0-or-later
from warnings import warn 

import numpy as np

import gmsh

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

def read_mesh(meshfile: str) -> np.ndarray:
    """
    Read a Gmsh mesh file and return element coordinates, node coordinates,
    and element indices grouped by physical group. The mesh must contain 
    exactly one element type in the topological dimension inferred from the 
    node coordinates.

    Parameters
    ----------
    meshfile : str
        Path to the mesh file readable by Gmsh.

    Returns
    -------
    xe : np.ndarray
        Element coordinates with shape ``(nel, n_nodes, ndim)``.
    coords : np.ndarray
        Array of all node coordinates with shape ``(n_nodes, 3)``.
    phys_groups : dict[str, np.ndarray]
        Mapping from physical-group name to the element indices belonging to
        that group. If a group has no name, ``group_<tag>`` is used.
    """
    # start gmsh
    gmsh.initialize()
    # read file
    gmsh.open(meshfile)
    # get nodes and node coordinates
    nodeTags, coords, nodeParams = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords).reshape(-1, 3)
    # infer dimensionality of mesh
    dim = 3 - np.sum(np.all(np.isclose(coords,0),axis=0))
    # get element information
    eltypes, eltags, elnodetags = gmsh.model.mesh.getElements(dim)
    # check if multiple element types used. if yes, raise error
    if len(eltypes) != 1:
        gmsh.finalize()
        raise RuntimeError("mesh must contain exactly one element type in the chosen dimension")
    #  
    _, _, _, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(eltypes[0])
    # find node indices corresponding to the elements node tags
    perm = np.argsort(nodeTags)
    pos = np.searchsorted(nodeTags[perm], elnodetags)
    # create element-wise node coordinates
    xe = coords[perm[pos], :dim].reshape(-1, n_nodes, dim)
    # map global element tags -> element index
    eltags = np.asarray(eltags[0])
    perm_el = np.argsort(eltags)
    sorted_eltags = eltags[perm_el]    
    # loop over physical groups
    phys_groups = {}
    for dim_pg, tag_pg in gmsh.model.getPhysicalGroups():
        #
        name = gmsh.model.getPhysicalName(dim_pg, tag_pg) or f"group_{tag_pg}"
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
    
        phys_groups[name] = np.asarray(idx, dtype=int)
    #
    gmsh.finalize()
    return xe, coords, phys_groups

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