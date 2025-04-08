import sympy as sp

if __name__ == "__main__":
    

    # Elastic moduli
    E1, E2, E3 = sp.symbols('E1 E2 E3')
    # Poisson's ratios (not assuming symmetry yet)
    nu12, nu13, nu23 = sp.symbols('nu12 nu13 nu23')
    # Shear moduli
    G12, G23, G31 = sp.symbols('G12 G23 G31')
    
    # Compliance matrix S (6x6)
    S = sp.Matrix([[1/E1, -nu12/E2, -nu13/E3, 0, 0, 0],
                   [-nu12/E1, 1/E2, -nu23/E3, 0, 0, 0],
                   [-nu13/E1, -nu23/E2, 1/E3, 0, 0, 0],
                   [0, 0, 0, 1/G23, 0, 0],
                   [0, 0, 0, 0, 1/G31, 0],
                   [0, 0, 0, 0, 0, 1/G12]])
    # Invert S to get stiffness matrix C
    C = S.inv()
    # simplify
    C = C.applyfunc(sp.simplify)
    # Optional: apply substitutions to enforce symmetry
    # C_simplified = C_simplified.subs(substitutions)
    print(C)
