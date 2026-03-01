# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import symbols
from symfem.functions import MatrixFunction

from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.huhu import huhu_engdensity, huhu_tangent

if __name__ == "__main__":
    #
    print("Energy density")
    for dim in range(3,4):
        print(str(dim)+"D")
        print(convert_to_code(MatrixFunction([[huhu_engdensity(u=None,
                                                               order=1,
                                                               ndim=dim, 
                                                               parallel=True)._f]]),
                              matrices=[],
                              vectors=["l","g"],
                              vectors_ele=["u"]),
              "\n")
    #
    import sys 
    sys.exit()
    print("Tangent without exponential")
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(huhu_tangent(u=None,
                                           ndim=dim,
                                           a=None, 
                                           do_integral=True),
                              matrices=[],vectors=["l","g"]),"\n")
    #
    print("Tangent with exponential: Picard iteration")
    for dim in range(2,3):
        print(str(dim)+"D")
        Ke, fe = huhu_tangent(u=None,
                              ndim=dim,
                              a=symbols("a"),
                              do_integral=True,
                              method="picard")
        print(convert_to_code(Ke,
                              matrices=[],
                              vectors=["kr",
                                       "a","l","g"], 
                              vectors_ele=["u"]),"\n")
    
    #
    import sys 
    sys.exit()
    print("Tangent with exponential: Newton iteration")
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(huhu_tangent(u=None,
                                           ndim=dim,
                                           a=symbols("a"), 
                                           do_integral=True, 
                                           mode="newton"),
                              matrices=[],
                              vectors=["l","g"], 
                              vectors_ele=["u"]),"\n")