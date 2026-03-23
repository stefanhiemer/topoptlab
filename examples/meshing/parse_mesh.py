from typing import Union

import numpy as np

import gmsh
from topoptlab.geometry_parser import mesh_cadfile, cad_to_mesh, mesh_to_xe

def create_testmesh(N: np.ndarray = np.ndarray([12,4,2]),
                    L: Union[None,np.ndarray] = None, 
                    mesh_file: Union[None,str] = None, 
                    gui: bool = False) -> None:
    """
    Create a simple mesh file of a box for testing.

    Parameters
    ----------
    N : np.ndarray
        number of elements in each direction.
    L : None or np.ndarray
        length in each direction. If None, 
    mesh_file : None or str
        path to mesh file. If None, simply called 'mesh-{dim}.msh'.
    gui : bool
        show the gmsh GUI to inspect the mesh.

    Returns
    -------
    None
        DESCRIPTION.

    """
    
    #
    dim = N.shape[0]
    #
    if L is None:
        L = N.copy()-1
    # 
    if mesh_file is None:
        mesh_file = f"mesh-{dim}.msh"
    # 
    gmsh.initialize()
    gmsh.model.add("structured_rect_or_box")
    if dim == 2:
        # create rectangle
        tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L[0], L[1])
        gmsh.model.occ.synchronize()
        # get boundary curves of the surface
        boundary = gmsh.model.getBoundary([(2, tag)], oriented=False)
        curves = [c[1] for c in boundary]
        # identify horizontal/vertical lines by bounding box
        for c in curves:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
            if abs(ymax - ymin) < 1e-6: 
                gmsh.model.mesh.setTransfiniteCurve(c, N[0] + 1)
            elif abs(xmax - xmin) < 1e-6: 
                gmsh.model.mesh.setTransfiniteCurve(c, N[1] + 1)
        # structured transfinite mesh
        gmsh.model.mesh.setTransfiniteSurface(tag)
        # quadrilateral elements instead of triangles
        gmsh.model.mesh.setRecombine(2, tag)  
    elif dim == 3:
        # create box
        tag = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L[0], L[1], L[2])
        gmsh.model.occ.synchronize()
        # get all edges of the volume
        surfaces = gmsh.model.getBoundary([(3, tag)], 
                                          oriented=False)
        edges = gmsh.model.getBoundary(surfaces, 
                                       oriented=False, 
                                       combined=False)
        curves = sorted(set(c[1] for c in edges))
        # assign transfinite divisions by edge direction
        for c in curves:
            #
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, 
                                                                           c)
            #
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin
            #
            if dx > dy and dx > dz:
                gmsh.model.mesh.setTransfiniteCurve(c, N[0] + 1)
            elif dy > dx and dy > dz:
                gmsh.model.mesh.setTransfiniteCurve(c, N[1] + 1)
            else:
                gmsh.model.mesh.setTransfiniteCurve(c, N[2] + 1)
        # set all boundary faces transfinite/recombined
        for sdim, stag in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(stag)
            gmsh.model.mesh.setRecombine(2, stag)
        #
        gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.model.mesh.setRecombine(3, tag)
    else:
        gmsh.finalize()
        raise ValueError("dim must be 2 or 3")
    # mesh
    gmsh.model.mesh.generate(dim)
    gmsh.write(mesh_file)

    # launch GUI
    if gui:
        gmsh.fltk.run()
    # close file
    gmsh.finalize()
    return
    
def create_testcad(L: np.ndarray, 
                   cad_file: Union[None,str] = None, 
                   gui: bool = False) -> None:
    """
    Create a simple step file of a box for testing.

    Parameters
    ----------
    L : None or np.ndarray
        length in each direction. If None, 
    cad_file : None or str
        path to mesh file. If None, simply called 'mesh-{dim}.step'.
    gui : bool
        show the gmsh GUI to inspect the geometry.

    Returns
    -------
    None
        DESCRIPTION.

    """
    #
    dim = L.shape[0]
    # 
    if cad_file is None:
        cad_file = f"mesh-{dim}.step"
    # 
    gmsh.initialize()
    gmsh.model.add("rect_or_box")
    if dim == 2:
        # create rectangle
        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L[0], L[1])
    elif dim == 3:
        # create box
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, L[0], L[1], L[2])
    else:
        gmsh.finalize()
        raise ValueError("dim must be 2 or 3")
    gmsh.model.occ.synchronize()
    gmsh.write(cad_file)

    # launch GUI
    if gui:
        gmsh.fltk.run()
    # close file
    gmsh.finalize()
    return

if __name__ == "__main__":
    #
    L = np.array([12,4])
    gui=True
    #
    create_testcad(L=L, 
                   gui=gui)
    #cad_to_mesh(file="mesh-{0}.step".format(L.shape[0]),
    #            mesh_dim = L.shape[0], 
    #            mesh_file = "mesh-{0}.msh".format(L.shape[0]),
    #            transfinite_transform = True,
    #            npoints = 10, 
    #            check_rect = True,
    #            check_hex = True,
    #            show_gui = gui)
    
    mesh_cadfile(cad_file="mesh-{0}.step".format(L.shape[0]), 
                 output_file="mesh-{0}.msh".format(L.shape[0]), 
                 etype="quad",
                 show_gui=gui)
    print(mesh_to_xe("mesh-{0}.msh".format(L.shape[0])))