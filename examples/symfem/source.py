# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.source import source
from topoptlab.symbolic.code_conversion import convert_to_code

if __name__ == "__main__":


    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(source(ndim = dim),
                              vectors=["b","l"]))
