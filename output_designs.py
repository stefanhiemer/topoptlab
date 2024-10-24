import numpy as np

from meshio import Mesh
from meshio.xdmf import TimeSeriesWriter

def threshold(xPhys,
              volfrac):
    """
    Threshold grey scale design to black and white design.

    Parameters
    ----------
    xPhys : np.array, shape (nel)
        element densities for topology optimization used for scaling the 
        material properties. 
    volfrac : float
        volume fraction.

    Returns
    -------
    xPhys : np.array, shape (nel)
        thresholded element densities for topology optimization used for scaling the 
        material properties. 

    """
    indices = np.flip(np.argsort(xPhys))
    vt = np.floor(volfrac*xPhys.shape[0]).astype(int)
    xPhys[indices[:vt]] = 1.
    xPhys[indices[vt:]] = 0.
    print("Thresholded Vol.: {0:.3f}".format(vt/xPhys.shape[0]))
    return xPhys

def export_vtk(filename, 
               nelx,nely, 
               xPhys,
               x=None,
               u=None, f=None,
               xTilde=None,
               volfrac=None):
    
    # construct node positions for meshio 
    _x,_y = np.meshgrid(np.linspace(0,nelx,nelx+1),
                        np.linspace(0,nely,nely+1)[-1::-1])
    points = np.column_stack((_x.flatten("F"),
                              _y.flatten("F")))
    print(points[:])
    # insert data for nodes
    node_data = {}
    if not u is None:
        # takes care of multiple load cases
        for i in np.arange(u.shape[1]):
            node_data.update({f"u{i}": u[:,i].reshape(points.shape[0],
                                         int(u[:,i].shape[0]/points.shape[0]))})
    if not f is None:
        # takes care of multiple load cases
        for i in np.arange(u.shape[1]):
            node_data.update({f"f{i}": f[:,i].reshape(points.shape[0],
                                             int(f[:,i].shape[0]/points.shape[0]))})
    #
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMat = np.column_stack((n1+1, n2+1, n2, n1))
    # insert data for elements
    el_data = {}
    el_data.update({"xPhys": [xPhys]})
    if not x is None:
        el_data.update({"x": [x]})
    if not xTilde is None:
        el_data.update({"xTilde": [xTilde]})
    if not volfrac is None:
        el_data.update({"xThresh": [threshold(xPhys,volfrac)]})
    #
    Mesh(points,
         [("quad", edofMat)],
         point_data=node_data, 
         cell_data=el_data).write(filename)
    return
