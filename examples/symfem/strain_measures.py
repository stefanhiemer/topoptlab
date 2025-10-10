# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import simplify

from topoptlab.symbolic.voigt import convert_from_voigt
from topoptlab.symbolic.strain_measures import def_grad,disp_grad, eng_strain
from topoptlab.symbolic.utils import is_equal

if __name__ == "__main__":
    #
    ndim = 3
    #
    eps = convert_from_voigt(eng_strain(ndim = ndim))
    #print("engineering strain: ", eps, "\n\n")
    #
    H = disp_grad(ndim = ndim)
    #print("displacement gradient: ", H, "\n\n")
    #
    F = def_grad(ndim = ndim)
    #print("deformation gradient: ", F, "\n\n")
    #
    #print()
    #
    test =  H + H.transpose() 
    for i in range(ndim):
        print(i,i)
        print(eps[i,i] ==  test[i,i]/2 )
        print()
        for j in range(i+1,ndim):
            print(i,j)
            print( is_equal(eps[i,j].as_sympy(),
                            simplify( test[i,j].as_sympy()) ))
            print()
