# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.mass_matrix import mass 

if __name__ == "__main__":


    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(mass(scalarfield=False,
                                   ndim = dim),vectors=["l"]),"\n")
