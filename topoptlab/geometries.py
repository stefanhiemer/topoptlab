import numpy as np

def sphere(nelx, nely, center, radius, fill_value=1):
    """
    Create element flags for a sphere located at the specified center with the
    specified radius.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    center : list or tuple or np.ndarray
        coordinates of sphere center.
    radius : float
        sphere radius.
    fill_value: int
        value that is prescribed to elements within sphere.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)

    """
    n = nelx*nely
    el = np.arange(n, dtype=np.int32)
    i,j = np.divmod(el,nely)
    mask = (i-center[0])**2 + (j-center[1])**2 <= radius**2
    #
    el_flags = np.zeros(n,dtype=np.int32)
    el_flags[mask] = fill_value
    return el_flags

def ball(nelx, nely, nelz, center, radius, fill_value=1):
    """
    Create element flags for a ball located at the specified center with the
    specified radius.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
        number of elements in z direction.
    center : list or tuple or np.ndarray
        coordinates of sphere center.
    radius : float
        sphere radius.
    fill_value: int
        value that is prescribed to elements within ball.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely*nelz)

    """
    n = nelx*nely*nelz
    el = np.arange(n, dtype=np.int32)
    k,ij = np.divmod(el,nelx*nely)
    i,j = np.divmod(ij,nely)
    mask = (i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2 <= radius**2
    #
    el_flags = np.zeros(n, dtype=np.int32)
    el_flags[mask] = fill_value
    return el_flags

def bounding_rectangle(nelx,nely,faces=["b","t","r","l"]):
    """
    Create element flags for a bounding box of one element thickness. It is
    possible to draw only specified faces of the bounding box.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    faces : list of str
        which faces of bounding box are supposed to be drawn. Possible
        values are "b" for bottom, "t" for top, "l" for left and "r" for right.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)

    """
    # collect indices
    indices = []
    # append corner indices
    if "t" in faces or "l" in faces:
        indices.append(0)
    if "t" in faces or "r" in faces:
        indices.append((nelx-1)*nely)
    if "b" in faces or "l" in faces:
        indices.append(nely - 1)
    if "b" in faces or "r" in faces:
        indices.append(nelx*nely-1)
    # append faces without corner indices
    if "t" in faces:
        indices.append(np.arange(nely,(nelx-1)*nely,nely))
    if "b" in faces:
        indices.append(np.arange(nely-1,nelx*nely,nely))
    if "l" in faces:
        indices.append(np.arange(1,nely-1))
    if "r" in faces:
        indices.append(np.arange((nelx-1)*nely + 1,nelx*nely-1))
    #
    indices = np.hstack(indices)
    el_flags = np.zeros(nelx*nely,dtype=int)
    # set to active
    el_flags[indices] = 2
    return el_flags
