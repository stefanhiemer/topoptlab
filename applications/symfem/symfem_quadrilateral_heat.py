from sympy import symbols
from symfem.functions import VectorFunction
import symfem
from symfem.symbols import x

if __name__ == "__main__":
    # Define the vertived and triangles of the mesh
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    rectangle = [0, 1, 3, 2]
    # Create a Lagrange element
    element = symfem.create_element("quadrilateral", "Lagrange", 1)
    # Create a matrix of zeros with the correct shape
    matrix = [[0 for i in range(4)] for j in range(4)]
    # anisotropic heat conductivity or equivalent
    k11,k12,k21,k22 = symbols("k11 k12 k21 k22")
    # Get the vertices of the triangle
    vs = tuple(vertices[i] for i in rectangle)
    # Create a reference cell with these vertices: this will be used
    # to compute the integral over the triangle
    ref = symfem.create_reference("quadrilateral", vertices=vs)
    # Map the basis functions to the cell
    basis = element.map_to_cell(vs)
    # 
    for test_i, test_f in zip(rectangle, basis):
        # 
        for trial_i, trial_f in zip(rectangle, basis):
            # Compute the integral of grad(u)-dot-grad(v) for each pair of basis
            # functions u and v. The second input (x) into `ref.integral` tells
            # symfem which variables to use in the integral.
            #integrand = test_f.grad(2).dot(trial_f.grad(2))
            f = test_f.grad(2) 
            g = trial_f.grad(2)
            f = VectorFunction([f.dot(VectorFunction([k11,k21])),
                                f.dot(VectorFunction([k12,k22]))])
            integrand = f.dot(g)
            print(integrand)
            matrix[test_i][trial_i] += integrand.integral(ref, x)
    
    print(matrix)