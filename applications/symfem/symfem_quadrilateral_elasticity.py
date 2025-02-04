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
    matrix = [[0 for i in range(8)] for j in range(8)]
    # anisotropic stiffness tensor or equivalent
    c11,c12,c13 = symbols("c11 c12 c13")
    c21,c22,c23 = symbols("c21 c22 c23")
    c31,c32,c33 = symbols("c31 c32 c33")
    # Get the vertices of the triangle
    vs = tuple(vertices[i] for i in rectangle)
    # Create a reference cell with these vertices: this will be used
    # to compute the integral over the triangle
    ref = symfem.create_reference("quadrilateral", vertices=vs)
    # Map the basis functions to the cell
    basis = element.map_to_cell(vs)
    #print(basis)
    # b.T matrix for test functions
    bt_test =  [[0 for i in range(3)] for j in range(8)]       
    for test_i, test_f in zip(rectangle, basis):
        # 
        f = test_f.grad(2) 
        # tension
        for i in range(2):
            bt_test[2*test_i+i][i] = f[i]
        # shear
        bt_test[2*test_i][2] = f[1]
        bt_test[2*test_i + 1][2] = f[0]
    # b matrix for trial functions
    b_trial = [[0 for i in range(8)] for j in range(3)] 
    for trial_i, trial_f in zip(rectangle, basis):
        # 
        g = trial_f.grad(2)
        # tension
        for i in range(2):
            b_trial[i][2*trial_i+i] = g[i]
        #
        b_trial[2][2*trial_i] = g[1]
        b_trial[2][2*trial_i + 1] = g[0]
    # convert to Vector functions
    bt_test = [VectorFunction(b) for b in bt_test]
    b_trial = [VectorFunction([b[i] for b in b_trial]) for i in range(8)]
    for i,btest in enumerate(bt_test):
        #
        btest = VectorFunction([btest.dot(VectorFunction([c11,c21,c31])),
                                btest.dot(VectorFunction([c12,c22,c32])),
                                btest.dot(VectorFunction([c13,c23,c33]))])
        for j,btrial in enumerate(b_trial):
            integrand = btest.dot(btrial)
            matrix[i][j] += integrand.integral(ref, x)
    print(matrix)