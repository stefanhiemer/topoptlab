# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple

import numpy as np
from scipy.signal import sawtooth
from matplotlib.patches import Polygon,Ellipse
from matplotlib.figure import Figure 
from matplotlib.axes import Axes

def spring(x0: float, y0: float,
           num_coils: int = 3,
           coil_width: float = 0.02,
           coil_length: float = 0.4,
           points_per_coil: int = 9) -> Tuple[np.ndarray,np.ndarray]:
    """
    Draw a spring for a sketch.

    Parameters
    ----------
    x0 : float
        x coordinate of bottom of spring.
    y0 : float
        y coordinate of bottom spring.
    num_coils : int 
        number of coils 
    coil_width : float 
        coil width
    coil_length : float 
        coil length
    points_per_coil : int
        points drawn per coil

    Returns
    -------
    x : np.ndarray shape (num_coils * points_per_coil + 1)
        x coordinates of spring.
    y : np.ndarray shape (num_coils * points_per_coil + 1)
        y coordinates of spring.
    """
    #
    t = np.linspace(np.pi/2, np.pi/2 + 2 * np.pi * num_coils, 
                    num_coils * points_per_coil + 1)
    #
    x = coil_width * sawtooth(t,width=0.5)
    y = np.linspace(0, coil_length, t.shape[0])
    return x0+x,y0+y

def hinged_support(x0: float, y0: float,
                   ax: Axes, fig: Figure,
                   triangle_width: float = 1.,
                   radius: float = 0.08) -> None:
    """
    Draw a hinged support for a sketch.

    Parameters
    ----------
    x0 : float
        x coordinate of supported point.
    y0 : float
        y coordinate of supported point.
    num_coils : int 
        number of coils 
    coil_width : float 
        coil width
    coil_length : float 
        coil length
    points_per_coil : int
        points drawn per coil

    Returns
    -------
    None
    """
    # get aspect ratio
    w,h = fig.get_figwidth(),fig.get_figheight()
    ratio = w/h
    # triangle
    triangle_coords = np.array([[0., 0.], [-0.5, -1.], [0.5, -1.]])
    triangle_coords = triangle_coords*triangle_width * np.array( [[1/ratio,ratio]] ) 
    triangle_coords += np.array([[x0,y0]])
    triangle = Polygon(triangle_coords,
                       closed=True, fill=False, edgecolor='gray', 
                       linewidth=2.) 
    ax.add_patch(triangle)
    # circle
    circle = Ellipse( xy=(x0,y0), width=radius/ratio, height=radius*ratio,
                     fill=True, edgecolor='gray', facecolor="white",
                     linewidth=2.)
    ax.add_patch(circle)
    # lines under the triangle
    for i in range(-5, 6, 2):
        ax.plot(np.array([i * 0.1, (i-1 ) * 0.1])*triangle_width/ratio + x0, 
                (np.array([-1, -1.2])*triangle_width * ratio)+y0, 
                color='gray', 
                linewidth=2.)
    return