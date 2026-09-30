# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.volume_integral import volume_integral

if __name__ == "__main__":

    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(volume_integral(ndim = dim), 
              vectors=["u","l"]))
