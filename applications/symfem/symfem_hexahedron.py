from sympy import symbols
import symfem
from symfem.symbols import x
from symfem.functions import VectorFunction

if __name__ == "__main__":
    
    # Define the vertived and triangles of the mesh
    vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    rectangular_cuboid = [0, 1, 3, 2,
                          4, 5, 7, 6]
    # anisotropic heat conductivity or equivalent
    k11,k12,k13 = symbols("k11 k12 k13")
    k21,k22,k23 = symbols("k21 k22 k23")
    k31,k32,k33 = symbols("k31 k32 k33")
    # Create a matrix of zeros with the correct shape
    matrix = [[0 for i in range(8)] for j in range(8)]
    # Create a Lagrange element
    element = symfem.create_element("hexahedron", "Lagrange", 1)
    #
    # Get the vertices of the element
    vs = tuple(vertices[i] for i in rectangular_cuboid)
    # Create a reference cell with these vertices: this will be used
    # to compute the integral over the triangle
    ref = symfem.create_reference("hexahedron", vertices=vs)
    # Map the basis functions to the cell
    basis = element.map_to_cell(vs)
    for test_i, test_f in zip(rectangular_cuboid, basis):
        for trial_i, trial_f in zip(rectangular_cuboid, basis):
            # Compute the integral of grad(u)-dot-grad(v) for each pair of basis
            # functions u and v. The second input (x) into `ref.integral` tells
            # symfem which variables to use in the integral.
            f = test_f.grad(3) 
            g = trial_f.grad(3)
            #
            f = VectorFunction([f.dot(VectorFunction([k11,k21,k31])),
                                f.dot(VectorFunction([k12,k22,k32])),
                                f.dot(VectorFunction([k13,k23,k33]))])
            integrand = f.dot(g)
            print(integrand)
            matrix[test_i][trial_i] += integrand.integral(ref, x)
    
    print(matrix)