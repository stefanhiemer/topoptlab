# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple, Union

from sympy import symbols
from symfem.functions import MatrixFunction

def rotation_matrix(ndim: int, mode: Union[None,str] = None
                    ) -> MatrixFunction:
    """
    rotation matrix around y and z axis with angles phi (y axis) and theta
    (z axis).

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    mode : str or None
        Either None or "voigt". If None, returns the standard rotation matrix.
        Either None or "voigt". If None, returns the standard rotation matrix.
        If "voigt" rotation matrix for 2nd rank tensors in Voigt notation
        ("Voigt vectors")  or 4th rank tensors ("Voigt matrices").

    Returns
    -------
    R : symfem.functions.MatrixFunction, shape (ndim,ndim) or
        ((ndim**2 + ndim) /2,(ndim**2 + ndim) /2)
        rotation matrix.

    """
    from sympy.functions.elementary.trigonometric import sin,cos
    # introduce angle variables
    if ndim == 1:
        pass
    elif ndim == 2:
        theta = symbols("theta")
    elif ndim == 3:
        theta,phi = symbols("theta phi")
    else:
        raise ValueError("ndim has to be integer and between 1 and 3.")
    # standard rotation matrix
    if mode is None:
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            theta = symbols("theta")
            R =  MatrixFunction([[cos(theta),-sin(theta)],
                                 [sin(theta),cos(theta)]])
        elif ndim == 3:
            theta,phi = symbols("theta phi")
            R = MatrixFunction([[cos(theta)*cos(phi),-sin(theta),cos(theta)*sin(phi)],
                                [sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi)],
                                [-sin(phi),0,cos(phi)]])
        return R
    elif mode == "voigt":
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            R = MatrixFunction([[cos(theta)**2, sin(theta)**2, -sin(2*theta)/2],
                                [sin(theta)**2, cos(theta)**2,  sin(2*theta)/2],
                                [ sin(2*theta), -sin(2*theta),    cos(2*theta)]])
        elif ndim == 3:
            R = MatrixFunction([[cos(phi)**2*cos(theta)**2,
                                 sin(theta)**2,
                                 sin(phi)**2*cos(theta)**2,
                                 0,
                                 sin(phi)*cos(phi)*cos(theta)**2,
                                 0],
                                [sin(theta)**2*cos(phi)**2,
                                 cos(theta)**2,
                                 sin(phi)**2*sin(theta)**2,
                                 0,
                                 sin(phi)*sin(theta)**2*cos(phi),
                                 0],
                                [sin(phi)**2,
                                 0,
                                 cos(phi)**2,
                                 0,
                                 -sin(2*phi)/2,
                                 0],
                                [-cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                 0,
                                 cos(2*phi - theta)/2 - cos(2*phi + theta)/2,
                                 0,
                                 -sin(2*phi - theta)/2 + sin(2*phi + theta)/2,
                                 0],
                                [-sin(2*phi - theta)/2 - sin(2*phi + theta)/2,
                                 0,
                                 sin(2*phi - theta)/2 + sin(2*phi + theta)/2,
                                 0,
                                 cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                 0],
                                [2*sin(theta)*cos(phi)**2*cos(theta),
                                 -sin(2*theta),
                                 2*sin(phi)**2*sin(theta)*cos(theta),
                                 0,
                                 cos(2*phi - 2*theta)/4 - cos(2*phi + 2*theta)/4,
                                 0]])
        return R

def rotation_matrix_dangle(ndim: int, mode: Union[None,str] = None
                           ) -> Tuple[MatrixFunction,MatrixFunction]:
    """
    1st derivative of rotation matrix around y and z axis with angles phi
    (y axis) and theta (z axis).

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    mode : str or None
        Either None or "voigt". If None, returns derivatives of the standard
        rotation matrix. If "voigt" rotation matrix for 2nd rank tensors in
        Voigt notation ("Voigt vectors")  or 4th rank tensors
        ("Voigt matrices").

    Returns
    -------
    dRdtheta : symfem.functions.MatrixFunction, shape (ndim,ndim) or
        ((ndim**2 + ndim) /2,(ndim**2 + ndim) /2)
        1st derivative of rotation matrix with regards to theta.
    dRdphi : symfem.functions.MatrixFunction, shape (ndim,ndim) or
        ((ndim**2 + ndim) /2,(ndim**2 + ndim) /2)
        1st derivative of rotation matrix with regards to phi.


    """
    from sympy.functions.elementary.trigonometric import sin,cos
    # introduce angle variables
    if ndim == 1:
        pass
    elif ndim == 2:
        theta = symbols("theta")
    elif ndim == 3:
        theta,phi = symbols("theta phi")
    else:
        raise ValueError("ndim has to be integer and between 1 and 3.")
    # standard rotation matrix
    if mode is None:
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            theta = symbols("theta")
            dRdtheta =  MatrixFunction([[-sin(theta), -cos(theta)],
                                        [cos(theta), -sin(theta)]])
            return dRdtheta

        elif ndim == 3:
            theta,phi = symbols("theta phi")
            dRdtheta = MatrixFunction([[-sin(theta)*cos(phi), -cos(theta), -sin(phi)*sin(theta)],
                                       [cos(phi)*cos(theta), -sin(theta),  sin(phi)*cos(theta)],
                                       [0, 0, 0]])
            dRdphi = MatrixFunction([[-sin(phi)*cos(theta), 0, cos(phi)*cos(theta)],
                                     [-sin(phi)*sin(theta), 0, sin(theta)*cos(phi)],
                                     [-cos(phi), 0, -sin(phi)]])

            return dRdtheta,dRdphi
    elif mode == "voigt":
        if ndim == 1:
            R =  MatrixFunction([[1]])
        elif ndim == 2:
            dRdtheta = MatrixFunction([[-sin(2*theta), sin(2*theta), -cos(2*theta)],
                                       [sin(2*theta),   -sin(2*theta),    cos(2*theta)],
                                       [2*cos(2*theta), -2*cos(2*theta), -2*sin(2*theta)]])
            return dRdtheta
        elif ndim == 3:
            dRdtheta = MatrixFunction([[-2*sin(theta)*cos(phi)**2*cos(theta),
                                        sin(2*theta),
                                        -2*sin(phi)**2*sin(theta)*cos(theta),
                                        0,
                                        -cos(2*phi - 2*theta)/4 + cos(2*phi + 2*theta)/4,
                                        0],
                                       [2*sin(theta)*cos(phi)**2*cos(theta),
                                        -sin(2*theta),
                                        2*sin(phi)**2*sin(theta)*cos(theta),
                                        0,
                                        cos(2*phi - 2*theta)/4 - cos(2*phi + 2*theta)/4,
                                        0],
                                       [0, 0, 0, 0, 0, 0],
                                       [-sin(2*phi - theta)/2 - sin(2*phi + theta)/2,
                                        0,
                                        sin(2*phi - theta)/2 + sin(2*phi + theta)/2,
                                        0,
                                        cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                        0],
                                       [cos(2*phi - theta)/2 - cos(2*phi + theta)/2,
                                        0,
                                        -cos(2*phi - theta)/2 + cos(2*phi + theta)/2,
                                        0,
                                        (2*sin(phi)**2 - 1)*sin(theta),
                                        0],
                                       [2*cos(phi)**2*cos(2*theta),
                                        -2*cos(2*theta),
                                        2*sin(phi)**2*cos(2*theta),
                                        0,
                                        sin(2*phi - 2*theta)/2 + sin(2*phi + 2*theta)/2,
                                        0]])
            dRdphi = MatrixFunction([[-2*sin(phi)*cos(phi)*cos(theta)**2,
                                      0,
                                      2*sin(phi)*cos(phi)*cos(theta)**2,
                                      0,
                                      cos(2*phi)*cos(theta)**2,
                                      0],
                                     [-2*sin(phi)*sin(theta)**2*cos(phi),
                                      0,
                                      2*sin(phi)*sin(theta)**2*cos(phi),
                                      0,
                                      sin(theta)**2*cos(2*phi),
                                      0],
                                     [sin(2*phi),
                                      0,
                                      -sin(2*phi),
                                      0,
                                      -cos(2*phi),
                                      0],
                                     [2*(2*sin(phi)**2 - 1)*sin(theta),
                                      0,
                                      -sin(2*phi - theta) + sin(2*phi + theta),
                                      0,
                                      -cos(2*phi - theta) + cos(2*phi + theta),
                                      0],
                                     [2*(2*sin(phi)**2 - 1)*cos(theta),
                                      0,
                                      cos(2*phi - theta) + cos(2*phi + theta),
                                      0,
                                      -sin(2*phi - theta) - sin(2*phi + theta),
                                      0],
                                     [-cos(2*phi - 2*theta)/2 + cos(2*phi + 2*theta)/2,
                                      0,
                                      cos(2*phi - 2*theta)/2 - cos(2*phi + 2*theta)/2,
                                      0,
                                      -sin(2*phi - 2*theta)/2 + sin(2*phi + 2*theta)/2,
                                      0]])
            return dRdtheta,dRdphi
        return R
