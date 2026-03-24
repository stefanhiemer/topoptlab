from typing import Union

import numpy as np

import gmsh
from topoptlab.geometry_parser import mesh_cadfile, cad_to_mesh, mesh_to_xe

def create_testmesh_physgroups(N: np.ndarray = np.array([12, 4, 2]),
                               L: Union[None, np.ndarray] = None,
                               mesh_file: Union[None, str] = None,
                               gui: bool = False, 
                               width: float = 0.2) -> None:
    """
    Create an L-shaped transfinite test mesh with 3 physical groups.
    """
    dim = N.shape[0]
    if L is None:
        L = N.copy() - 1
    if mesh_file is None:
        mesh_file = f"mesh-{dim}.msh"

    gmsh.initialize()
    gmsh.model.add("L_groups")

    if dim == 2:
        a1 = gmsh.model.occ.addRectangle(0, 0, 0,  
                                         L[0]*width, L[1]*width)
        a2 = gmsh.model.occ.addRectangle(L[0]*width, 0, 0, 
                                         L[0]*(1-width), L[1]*width)
        a3 = gmsh.model.occ.addRectangle(0, L[1]*width, 0, 
                                         L[0]*width, L[1]*(1-width))
        gmsh.model.occ.fragment([(2, a1)], [(2, a2), (2, a3)])
        gmsh.model.occ.synchronize()

        areas = [t for _, t in gmsh.model.getEntities(2)]
        for a in areas:
            curves = [c[1] for c in gmsh.model.getBoundary([(2, a)], oriented=False)]
            for c in curves:
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
                if abs(xmax - xmin) > abs(ymax - ymin):
                    gmsh.model.mesh.setTransfiniteCurve(c, N[0] // 2 + 1)
                else:
                    gmsh.model.mesh.setTransfiniteCurve(c, N[1] // 2 + 1)
            gmsh.model.mesh.setTransfiniteSurface(a)
            gmsh.model.mesh.setRecombine(2, a)

        for i, a in enumerate(areas, 1):
            pg = gmsh.model.addPhysicalGroup(2, [a])
            gmsh.model.setPhysicalName(2, pg, f"group_{i}")

    elif dim == 3:
        v1 = gmsh.model.occ.addBox(0,      0,      0, 
                                   L[0]*width, L[1]*width, L[2]*width)
        v2 = gmsh.model.occ.addBox(L[0]*width, 0,      0, 
                                   L[0]*(1-width), L[1]*width, L[2]*width)
        v3 = gmsh.model.occ.addBox(0,      L[1]*width, 0, 
                                   L[0]*width, L[1]*(1-width), L[2]*width)
        gmsh.model.occ.fragment([(3, v1)], [(3, v2), (3, v3)])
        gmsh.model.occ.synchronize()

        vols = [t for _, t in gmsh.model.getEntities(3)]
        for v in vols:
            surfaces = gmsh.model.getBoundary([(3, v)], oriented=False)
            edges = gmsh.model.getBoundary(surfaces, oriented=False, combined=False)
            curves = sorted(set(c[1] for c in edges))
            for c in curves:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, c)
                dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
                if dx > dy and dx > dz:
                    gmsh.model.mesh.setTransfiniteCurve(c, N[0] // 2 + 1)
                elif dy > dx and dy > dz:
                    gmsh.model.mesh.setTransfiniteCurve(c, N[1] // 2 + 1)
                else:
                    gmsh.model.mesh.setTransfiniteCurve(c, N[2])
            for _, s in surfaces:
                gmsh.model.mesh.setTransfiniteSurface(s)
                gmsh.model.mesh.setRecombine(2, s)
            gmsh.model.mesh.setTransfiniteVolume(v)
            gmsh.model.mesh.setRecombine(3, v)

        for i, v in enumerate(vols, 1):
            pg = gmsh.model.addPhysicalGroup(3, [v])
            gmsh.model.setPhysicalName(3, pg, f"group_{i}")

    else:
        gmsh.finalize()
        raise ValueError("dim must be 2 or 3")

    gmsh.model.mesh.generate(dim)
    gmsh.write(mesh_file)

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()

def create_testmesh(N: np.ndarray = np.array([12,4,2]),
                    L: Union[None,np.ndarray] = None, 
                    mesh_file: Union[None,str] = None, 
                    gui: bool = False) -> None:
    """
    Create a regular mesh file of a box for testing.

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

    """
    
    #
    dim = N.shape[0]
    print(N)
    print(dim)
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
                gmsh.model.mesh.setTransfiniteCurve(c, N[0])
            elif abs(xmax - xmin) < 1e-6: 
                gmsh.model.mesh.setTransfiniteCurve(c, N[1])
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
                gmsh.model.mesh.setTransfiniteCurve(c, N[0])
            elif dy > dx and dy > dz:
                gmsh.model.mesh.setTransfiniteCurve(c, N[1])
            else:
                gmsh.model.mesh.setTransfiniteCurve(c, N[2])
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
    N = np.array([12,12,12])
    L = N-1
    gui=False
    #
    #create_testcad(L=L, 
    #               gui=gui)
    #cad_to_mesh(file="mesh-{0}.step".format(L.shape[0]),
    #            mesh_dim = L.shape[0], 
    #            mesh_file = "mesh-{0}.msh".format(L.shape[0]),
    #            transfinite_transform = True,
    #            npoints = 10, 
    #            check_rect = True,
    #            check_hex = True,
    #            show_gui = gui)
    
    #mesh_cadfile(cad_file="mesh-{0}.step".format(L.shape[0]), 
    #             output_file="mesh-{0}.msh".format(L.shape[0]), 
    #             etype="quad",
    #             show_gui=gui)
    create_testmesh_physgroups(N = N, gui=True)
    xe, coords, phys_groups = mesh_to_xe("mesh-{0}.msh".format(L.shape[0]))
    for group in phys_groups:
        print(group)
        print(xe[phys_groups[group]].min(axis=(0,1)))
        print(xe[phys_groups[group]].max(axis=(0,1)))
        print()