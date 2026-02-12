# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import symbols
from symfem.functions import MatrixFunction

from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.huhu import huhu_engdensity, huhu_tangent

if __name__ == "__main__":
    #
    print("Energy density")
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(MatrixFunction([[huhu_engdensity(u=None,
                                                             ndim=dim, 
                                                             parallel=False)._f]]),
                              matrices=[],
                              vectors=["l","g"],
                              vectors_ele=["u"]),
              "\n")
    #
    print("Tangent without exponential")
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(huhu_tangent(u=None,
                                           ndim=dim,
                                           a=None),
                              matrices=[],vectors=["l","g"]),"\n")
    #
    print("Tangent with exponential: Newton iteration")
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(huhu_tangent(u=None,
                                           ndim=dim,
                                           a=symbols("a"), 
                                           mode="newton"),
                              matrices=[],vectors=["l","g"]),"\n")
    #
    print("Tangent with exponential: Picard iteration")
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(huhu_tangent(u=None,
                                           ndim=dim,
                                           a=symbols("a"),
                                           method="picard"),
                              matrices=[],vectors=["l","g"]),"\n")