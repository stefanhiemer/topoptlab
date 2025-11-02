# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.lin_elasticity import stiffness_matrix,strainforces
from topoptlab.symbolic.stiffness_tensors import stifftens_isotropic
from topoptlab.symbolic.code_conversion import convert_to_code 

if __name__ == "__main__":
    
    # stiffness matrix isotropic linear elasticity
    for dim in range(1,3):
        print(str(dim)+"D")
        print(convert_to_code(stiffness_matrix(c=stifftens_isotropic(ndim=dim,
                                                            plane_stress=True), 
                                               plane_stress=True, ndim=dim),
                              matrices=["c"],vectors=["l","g"]),"\n")
    # stiffness matrix anisotropic linear elasticity
    for dim in range(1,3):
        print(str(dim)+"D")
        print(convert_to_code(stiffness_matrix(ndim = dim, 
                                               c=None, 
                                               plane_stress=True),
                              matrices=["c"],vectors=["l","g"]),"\n")
    
    for dim in range(1,3):
        print(str(dim)+"D")
        print(convert_to_code(strainforces(c=stifftens_isotropic(ndim=dim,
                                                            plane_stress=True), 
                                           plane_stress=True, ndim=dim),
                              matrices=["c"],vectors=["l","g", "eps"]),"\n") 
    for dim in range(1,3):
        print(str(dim)+"D")
        print(convert_to_code(strainforces(c=None, 
                                           plane_stress=True, ndim=dim),
                              matrices=["c"],vectors=["l","g", "eps"]),"\n")