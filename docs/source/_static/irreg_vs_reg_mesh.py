import gmsh

def show_only(surface_tag):
    # Hide everything, then show only one surface
    all_surfs = gmsh.model.getEntities(2)       # all 2D surfaces
    gmsh.model.setVisibility(all_surfs, False)  # hide them all
    gmsh.model.setVisibility([(2, surface_tag)], True)  # show one

def structured():
    gmsh.initialize()
    gmsh.model.add("reg")
    
    # structured mesh
    rect1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    
    # get lines
    entities = gmsh.model.getEntities(1)
    lines1 = [e[1] for e in entities if e[0] == 1 and e[1] <= 4]
    
    # transfinite meshing (structured grid)
    for l in lines1:
        gmsh.model.mesh.setTransfiniteCurve(l, 10)  # 10 divisions
    gmsh.model.mesh.setTransfiniteSurface(rect1)
    gmsh.model.mesh.setRecombine(2, rect1)          # quadrilateral mesh
    
    # generate mesh and image
    gmsh.model.mesh.generate(2)
    gmsh.fltk.initialize()
    gmsh.write("reg-mesh.png")
    gmsh.finalize()
    return

def unstructured():
    gmsh.initialize()
    gmsh.model.add("irreg")

    # Build square manually from points with varying mesh sizes
    lcA, lcB, lcC, lcD = 0.02, 0.05, 0.15, 0.25  # characteristic lengths
    p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lcA)
    p2 = gmsh.model.occ.addPoint(1.0, 0.0, 0.0, lcB)
    p3 = gmsh.model.occ.addPoint(1.0, 1.0, 0.0, lcC)
    p4 = gmsh.model.occ.addPoint(0.0, 1.0, 0.0, lcD)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.synchronize()

    # Use default unstructured triangulation
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
    gmsh.option.setNumber("Mesh.Smoothing", 1)

    # Generate and save
    gmsh.model.mesh.generate(2)
    gmsh.fltk.initialize()
    gmsh.write("irreg-mesh.png")
    gmsh.finalize()
    return


if __name__ == "__main__":
    unstructured()