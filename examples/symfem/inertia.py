# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.symbolic.code_conversion import wrap_function
from topoptlab.symbolic.advection import advection

if __name__ == "__main__":

    #
    for dim in range(3,4):
        print(str(dim)+"D")
        if dim == 1:
            signature = "v: np.ndarray, l: np.ndarray = np.array([1.]), **kwargs: Any"
        elif dim == 2:
            signature = ("v: np.ndarray, l: np.ndarray = np.array([1.,1.]), "
                         "g: np.ndarray = np.array([0.]), **kwargs: Any")
        else:
            signature = ("v: np.ndarray, l: np.ndarray = np.array([1.,1.,1.]), "
                         "g: np.ndarray = np.array([0.,0.]), **kwargs: Any")
        print(wrap_function(name=f"lk_advection_{dim}d",
                            signature=signature,
                            matrix=advection(ndim = dim),
                            vectors=["v","l","g"]))
