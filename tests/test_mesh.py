from typing import Union

from numpy import arange,array,column_stack,prod
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

def create_testfile(N: array = array([60,20]),
                    L: Union[None,array] = None, 
                    mesh_file: Union[None,str] = None, 
                    gui: bool = False) -> None:
    import gmsh
    #
    dim = N.shape[0]
    #
    if L is None:
        L = N.copy()
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
        # Get boundary curves of the surface
        boundary = gmsh.model.getBoundary([(2, tag)], oriented=False)
        curves = [c[1] for c in boundary]

        # identify horizontal/vertical lines by bounding box
        for c in curves:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
            if abs(ymax - ymin) < 1e-12: 
                gmsh.model.mesh.setTransfiniteCurve(c, N[0] + 1)
            elif abs(xmax - xmin) < 1e-12: 
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
        surfaces = gmsh.model.getBoundary([(3, tag)], oriented=False)
        edges = gmsh.model.getBoundary(surfaces, oriented=False)
        curves = sorted(set(c[1] for c in edges))

        # assign transfinite divisions by edge direction
        for c in curves:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, 
                                                                           c)
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

        gmsh.model.mesh.setTransfiniteVolume(tag)

    else:
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
    
