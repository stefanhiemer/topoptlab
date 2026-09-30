# SPDX-License-Identifier: GPL-3.0-or-later

from topoptlab.symbolic.heatexpansion import heatexp_iso, heatexp_aniso
from topoptlab.symbolic.code_conversion import convert_to_code

if __name__ == "__main__":

    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(heatexp_iso(ndim = dim),
                              matrices=["c"],vectors=["l","g"]),"\n") 
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(heatexp_aniso(ndim = dim),
                              matrices=["c"],vectors=["l","g","a"]),"\n")
