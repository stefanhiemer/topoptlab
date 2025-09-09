# SPDX-License-Identifier: GPL-3.0-or-later
from symfem import create_element

from topoptlab.symbolic.cell import determine_nodeinds

if __name__ == "__main__":
    vertices = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1), 
                (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1),
                (0, 0, -1), 
                (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),
                (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0),
                (0, 0, 0), 
                (0, -1, 1), (1, 0, 1), (0, 1, 1), (-1, 0, 1), 
                (0, 0, 1))
    element = create_element("hexahedron", 
                             "Lagrange", 
                             order=2)
    basis_funcs = element.get_basis_functions()
    print(len(basis_funcs),len(vertices))
    print(determine_nodeinds(vertices, 
                             basis_funcs, 
                             ndim=len(vertices[0])))
