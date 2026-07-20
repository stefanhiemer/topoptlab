# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.lin_elasticity import stiffness_matrix ,strainforces
from topoptlab.symbolic.viscosity_tensor import shear_viscosity
from topoptlab.symbolic.code_conversion import convert_to_code 

if __name__ == "__main__":
    
    # 
    for dim in range(2,4):
        print(str(dim)+"D")
        print(convert_to_code(stiffness_matrix(c=shear_viscosity(ndim=dim), 
                                               ndim=dim),
                              matrices=["c"],
                              vectors=["l","g"]),"\n")
    
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(strainforces(c=shear_viscosity(ndim=dim),
                                           ndim=dim),
                              matrices=["c"],
                              vectors=["l","g", "eps"]),"\n") 