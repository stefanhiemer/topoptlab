from symfem import create_element

from topoptlab.symfem_utils import determine_nodeinds

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
