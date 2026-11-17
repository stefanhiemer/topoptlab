# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.buckling import buckling_matrix
from topoptlab.symbolic.matrix_utils import generate_constMatrix
from topoptlab.symbolic.stiffness_tensors import stifftens_isotropic
from topoptlab.symbolic.code_conversion import convert_to_code

if __name__ == "__main__":

    # buckling matrix isotropic linear elasticity in terms of displacements
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(buckling_matrix(ndim=dim,
                                              c=stifftens_isotropic(ndim=dim,
                                                                    plane_stress=True)),
                              matrices=["c"],
                              vectors=["l","g","u"]),"\n")
    # buckling matrix anisotropic linear elasticity in terms of displacements
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(buckling_matrix(ndim=dim,
                                              c=None),
                              matrices=["c"],
                              vectors=["l","g","u"]),"\n")
    # buckling matrix for constant stress in Voigt notation
    for dim in range(1,4):
        print(str(dim)+"D")
        sigma = generate_constMatrix(ncol=1,
                                     nrow=int((dim**2 + dim) /2),
                                     name="s")
        print(convert_to_code(buckling_matrix(ndim=dim,
                                              sigma=sigma),
                              vectors=["l","g","s"]),"\n")
