# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.monomial import monomial

if __name__ == "__main__":

    #
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(monomial(mononomial_order=3,
                                       scalarfield=True,
                                       ndim = dim),
                              vectors=["l"],
                              vectors_ele=["u"]),"\n")
