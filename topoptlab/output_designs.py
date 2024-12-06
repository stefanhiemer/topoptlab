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
    xThresh = np.zeros(xPhys.shape)
    xThresh[indices[:vt]] = 1.
    xThresh[indices[vt:]] = 0.
    print("Thresholded Vol.: {0:.3f}".format(vt/xThresh.shape[0]))
    return xThresh

def export_vtk(filename, 
               nelx,nely, 
               xPhys,
               x=None,
               u=None, f=None,
               u_bw=None,
               f_bw=None,
               xTilde=None,
               volfrac=None):
    """
    Export design to a vtk file for visualisation e. g. with Paraview.

    Parameters
    ----------
    filename : str
        filename without ".vtk" ending.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    xPhys : np.ndarray
        densities used to scale the material properties.
    x : np.ndarray, optional
        interemdiary densities as (usually) returned by the optimizer.
    u : np.ndarray, optional
        field variable (displacements, temperature) which is usually the left 
        hand side of the generic matrix problem Ku=f. The default is None.
    f : np.ndarray, optional
        flow variable (forces, heat sources/sinks) which is usually the right 
        hand side of the generic matrix problem Ku=f
    u_bw : np.ndarray, optional
        field variable of the final black/white design. The default is None.
    f_bw : np.ndarray, optional
        flow variable of the final black/white design. The default is None.
    xTilde : np.ndarray, optional
        interemdiary densities by the density filter. So far they only occur 
        if Haeviside projections are used. The default is None.
    volfrac : float, optional
        volume fraction. If not None, then also a thresholded designed is 
        stored. The default is None.

    Returns
    -------
    None.

    """
    
    # construct node positions for meshio 
    _x,_y = np.meshgrid(np.linspace(0,nelx,nelx+1),
                        np.linspace(0,nely,nely+1)[-1::-1])
    points = np.column_stack((_x.flatten("F"),
                              _y.flatten("F"))) 
    # insert data for nodes
    node_data = {}
    if not u is None:
        # takes care of multiple load cases
        for i in np.arange(u.shape[1]):
            node_data.update({f"u{i}": u[:,i].reshape(points.shape[0],
                                         int(u[:,i].shape[0]/points.shape[0]))})
    if not u_bw is None:
        # takes care of multiple load cases
        for i in np.arange(u_bw.shape[1]):
            node_data.update({f"u_bw{i}": u_bw[:,i].reshape(points.shape[0],
                                         int(u_bw[:,i].shape[0]/points.shape[0]))})
    if not f is None:
        # takes care of multiple load cases
        for i in np.arange(u.shape[1]):
            node_data.update({f"f{i}": f[:,i].reshape(points.shape[0],
                                             int(f[:,i].shape[0]/points.shape[0]))})
    if not f_bw is None:
        # takes care of multiple load cases
        for i in np.arange(f_bw.shape[1]):
            node_data.update({f"f_bw{i}": f_bw[:,i].reshape(points.shape[0],
                                             int(f_bw[:,i].shape[0]/points.shape[0]))})
    # assign node IDs to each cell. 
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    idMat = np.column_stack((n1+1, n2+1, n2, n1))
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
         [("quad", idMat)],
         point_data=node_data, 
         cell_data=el_data).write(filename+".vtk")
    return

def export_stl(filename, 
               nelx,nely, 
               xPhys,
               volfrac):
    """
    Export design to a stl file for 3D printing.

    Parameters
    ----------
    filename : str
        filename without ".vtk" ending.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    xPhys : np.ndarray
        densities used to scale the material properties.
    volfrac : float, optional
        volume fraction. If not None, then also a thresholded designed is 
        stored. The default is None.

    Returns
    -------
    None.

    """
    # threshold densities to get final design
    mask = (threshold(xPhys,volfrac) == 1)
    # construct node positions for meshio 
    _x,_y = np.meshgrid(np.linspace(0,nelx,nelx+1),
                        np.linspace(0,nely,nely+1)[-1::-1])
    points = np.column_stack((_x.flatten("F"),
                              _y.flatten("F"))) 
    # assign node IDs to each triangle cell.
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    idMat = np.column_stack((n1+1, n2+1, n2, n1))
    idMat = np.vstack((idMat[mask][:,[0,1,2]],
                       idMat[mask][:,[0,2,3]]))
    #
    Mesh(points,
         [("triangle", idMat)],
         point_data={}, 
         cell_data={}).write(filename+".stl")
    return
