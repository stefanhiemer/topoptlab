from sympy import symbols
from symfem.functions import VectorFunction
import symfem
from symfem.symbols import x

if __name__ == "__main__":
    # Define the vertived and triangles of the mesh
    vertices = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    rectangular_cuboid = [0, 1, 3, 2,
                          4, 5, 7, 6]
    # Create a Lagrange element
    element = symfem.create_element("hexahedron", "Lagrange", 1)
    # Create a matrix of zeros with the correct shape
    matrix = [[0 for i in range(24)] for j in range(24)]
    # anisotropic stiffness tensor or equivalent
    c11,c12,c13,c14,c15,c16 = symbols("c11 c12 c13 c14 c15 c16")
    c21,c22,c23,c24,c25,c26 = symbols("c21 c22 c23 c24 c25 c26")
    c31,c32,c33,c34,c35,c36 = symbols("c31 c32 c33 c34 c35 c36")
    c41,c42,c43,c44,c45,c46 = symbols("c41 c42 c43 c44 c45 c46")
    c51,c52,c53,c54,c55,c56 = symbols("c51 c52 c53 c54 c55 c56")
    c61,c62,c63,c64,c65,c66 = symbols("c61 c62 c63 c64 c65 c66")
    # Get the vertices of the triangle
    vs = tuple(vertices[i] for i in rectangular_cuboid)
    # Create a reference cell with these vertices: this will be used
    # to compute the integral over the triangle
    ref = symfem.create_reference("hexahedron", vertices=vs)
    # Map the basis functions to the cell
    basis = element.map_to_cell(vs)
    #print(basis)
    # b.T matrix for test functions
    bt_test =  [[0 for i in range(6)] for j in range(24)]       
    for test_i, test_f in zip(rectangular_cuboid, basis):
        # 
        f = test_f.grad(3) 
        # tension
        for i in range(3):
            bt_test[3*test_i+i][i] = f[i]
        # shear xy
        bt_test[3*test_i][-1] = f[1]
        bt_test[3*test_i + 1][-1] = f[0]
        # shear xz
        bt_test[3*test_i][-2] = f[2]
        bt_test[3*test_i + 2][-2] = f[0]
        # shear yz
        bt_test[3*test_i + 1][-3] = f[2]
        bt_test[3*test_i + 2][-3] = f[1]
    #print(bt_test)
    # b matrix for trial functions
    b_trial = [[0 for i in range(24)] for j in range(6)] 
    for trial_i, trial_f in zip(rectangular_cuboid, basis):
        # 
        g = trial_f.grad(3)
        # tension
        for i in range(3):
            b_trial[i][3*trial_i+i] = g[i]
        # shear xy
        b_trial[-1][3*trial_i] = g[1]
        b_trial[-1][3*trial_i + 1] = g[0]
        # shear xz
        b_trial[-2][3*trial_i] = g[2]
        b_trial[-2][3*trial_i + 2] = g[0]
        # shear yz
        b_trial[-3][3*trial_i + 1] = g[2]
        b_trial[-3][3*trial_i + 2] = g[1]
    #print(b_trial)
    #import sys 
    #sys.exit()
    # convert to Vector functions
    bt_test = [VectorFunction(b) for b in bt_test]
    b_trial = [VectorFunction([b[i] for b in b_trial]) for i in range(24)]
    #for b in bt_test[:3]:
    #    print(b[3:])
    #import sys 
    #sys.exit()
    #print(bt_test)
    #print(b_trial)
    #print(bt_test == b_trial)
    #import sys 
    #sys.exit()
    for i,btest in enumerate(bt_test):
        for j,btrial in enumerate(b_trial):
            print(i,j)
            #
            b = VectorFunction([btest.dot(VectorFunction([c11,c21,c31,c41,c51,c61])),
                                btest.dot(VectorFunction([c12,c22,c32,c42,c52,c62])),
                                btest.dot(VectorFunction([c13,c23,c33,c43,c53,c63])),
                                btest.dot(VectorFunction([c14,c24,c34,c44,c54,c64])),
                                btest.dot(VectorFunction([c15,c25,c35,c45,c55,c65])),
                                btest.dot(VectorFunction([c16,c26,c36,c46,c56,c66]))])
            integrand = b.dot(btrial)
            matrix[i][j] += integrand.integral(ref, x)
                
    print(matrix)