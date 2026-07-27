# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union
import numpy as np

from meshio import Mesh

from topoptlab.log_utils import BaseLogger, EmptyLogger

def threshold(xPhys: np.ndarray,
              volfrac: Union[float,np.ndarray], 
              logger: Union[None,BaseLogger] = None) -> np.ndarray:
    """
    Threshold grey scale design to black and white design.

    Parameters
    ----------
    xPhys : np.ndarray
        element densities for topology optimization used for scaling the 
        material properties. shape (nel,n_mat) 
    volfrac : float or np.ndarray
        volume fraction.
    logger : EmptyLogger or SimpleLogger
        logger for writing information to logfile.

    Returns
    -------
    xThresh : np.ndarray, shape (nel)
        thresholded element densities for topology optimization used for scaling the 
        material properties. 

    """
    #
    volfrac = np.atleast_1d(volfrac)
    xThresh = np.zeros(xPhys.shape, order="F")
    #
    vt_fracs = []
    for i in np.arange(xPhys.shape[1]):
        indices = np.flip(np.argsort(xPhys[:,i]))
        vt = np.floor(volfrac[i]*xPhys.shape[0]).astype(int)
        xThresh[indices[:vt],i] = 1.
        xThresh[indices[vt:],i] = 0.
        vt_fracs.append(vt/xThresh.shape[0])
    msg = "Thresholded Vol.: {}".format(", ".join(["{:.5f}".format(v)  for v in vt_fracs]))
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    return xThresh

def export_vtk(filename: str, 
               nelx: int, nely: int, 
               xPhys: Union[None,np.ndarray] = None,
               nelz: Union[None,int] = None,
               x: Union[None,np.ndarray] = None,
               u: Union[None,np.ndarray] = None, 
               f: Union[None,np.ndarray] = None,
               nodal_variables: Union[None,Dict[np.ndarray]] = None,
               element_variables: Union[None,Dict[np.ndarray]] = None,
               u_bw: Union[None,np.ndarray] = None,
               f_bw: Union[None,np.ndarray] = None,
               xTilde: Union[None,np.ndarray] = None,
               elem_size: Union[None,float, np.ndarray] = None,
               volfrac: Union[None,float] = None,
               stress_vm: Union[None,float, np.ndarray] = None,
               vectors: Union[None,np.ndarray] = None) -> None:
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
    nelz : int
        number of elements in z direction.
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
    elem_size : np.ndarray, float
        element size.
    volfrac : float, optional
        volume fraction. If not None, then also a thresholded designed is 
        stored. The default is None.
    stress_vm : np.ndarray, optional
        von Mises stress.
    vectors : np.ndarray
        vectors [cos(theta)cos(phi); sin(theta)cos(phi); -sin(phi)] used to show the 
        orientation arrow field in Paraview.

    Returns
    -------
    None.

    """
    
    # construct node positions for meshio 
    if nelz is None:
        _x,_y = np.meshgrid(np.linspace(0,nelx,nelx+1),
                            np.linspace(nely,0,nely+1))
        points = np.column_stack((_x.flatten("F"),
                                  _y.flatten("F"))) 
    else:
        #
        xcoords = np.linspace(0,nelx,nelx+1)
        ycoords = np.linspace(nely,0,nely+1)
        zcoords = np.linspace(0,nelz,nelz+1)
        #
        _x = np.tile(np.repeat(xcoords,nely+1),nelz+1).flatten()
        _y = np.tile(ycoords,(nelx+1)*(nelz+1)).flatten()
        _z = np.repeat(zcoords,(nelx+1)*(nely+1))
        #
        points = np.column_stack((_x,_y,_z)) 

    # Apply the physical spacing by element size
    if elem_size is not None:
        ndim = points.shape[1]
        if isinstance(elem_size, float):
            elem_size = np.full(ndim, elem_size, dtype=float)
        else:
            elem_size = np.asarray(elem_size, dtype=float)
        points = points * elem_size[None, :]
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
    if nodal_variables is not None:
        #
        for key in nodal_variables.keys():
            if nodal_variables[key] is None:
                continue
            for i in range(nodal_variables[key].shape[1]):
                node_data.update({f"{key}{i}": nodal_variables[key][:,i].reshape(points.shape[0],
                                             int(nodal_variables[key][:,i].shape[0]/points.shape[0]))})

    # assign node IDs to each cell. 
    if nelz is None:
        elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
        n1 = ((nely+1)*elx+ely).flatten()
        n2 = ((nely+1)*(elx+1)+ely).flatten()
        idMat = np.column_stack((n1+1, n2+1, n2, n1))
    else:
        elx = np.arange(nelx)[None,:,None]
        ely = np.arange(nely)[None,None,:]
        elz = np.arange(nelz)[:,None,None]
        n1 = ((nelx+1)*(nely+1)*elz + (nely+1)*elx + ely).flatten()
        n2 = ((nelx+1)*(nely+1)*elz + (nely+1)*(elx+1) + ely).flatten()
        n3 = ((nelx+1)*(nely+1)*(elz+1) + (nely+1)*elx + ely).flatten()
        n4 = ((nelx+1)*(nely+1)*(elz+1) + (nely+1)*(elx+1) + ely).flatten()
        idMat = np.column_stack((n1+1,n2+1,n2,n1,
                                 n3+1,n4+1,n4,n3))
    # insert data for elements
    el_data = {}
    if xPhys is not None: 
        el_data.update({"xPhys": [xPhys]})
    if x is not None:
        el_data.update({"x": [x]})
    if xTilde is not None:
        el_data.update({"xTilde": [xTilde]})
    if volfrac is not None and xPhys is not None:
        el_data.update({"xThresh": [threshold(xPhys,volfrac)]})
    if volfrac is not None and \
       element_variables is not None and \
       "xPhys" in element_variables.keys():
        el_data.update({"xThresh": [threshold(element_variables["xPhys"],volfrac)]})
    if stress_vm is not None:
        el_data["stress_vm"] = [np.asarray(stress_vm, dtype=float)]
    if vectors is not None:
        el_data["vectors"] = [np.asarray(vectors, dtype=float)]
    if element_variables is not None:
        #
        for key in element_variables.keys():
            if element_variables[key] is None:
                continue  

            if len(element_variables[key].shape)==2:
                for i in range(element_variables[key].shape[1]): 
                    el_data.update({f"{key}-{i}": [element_variables[key][:,i]]})
            elif len(element_variables[key].shape)==1:
                #
                if element_variables[key].dtype==np.int32:
                    element_variables[key] = element_variables[key].astype(np.int64)

                el_data.update({f"{key}": [element_variables[key]]})
            else:
                raise ValueError
    #
    if nelz is None:
        Mesh(points,
             [("quad", idMat)],
             point_data=node_data,
             cell_data=el_data).write(filename+".vtk")
    else:
        Mesh(points,
             [("hexahedron", idMat)],
             point_data=node_data,
             cell_data=el_data).write(filename+".vtk")
    return
