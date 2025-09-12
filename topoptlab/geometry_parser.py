# SPDX-License-Identifier: GPL-3.0-or-later
from warnings import warn 

import gmsh

def parse_cad_and_mesh(file: str, 
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