import numpy as np
from scipy.linalg import solve
from scipy.special import factorial

def bdf_coefficients(k):
    """
    Get coefficients for backward differentiation formula BDF. Shamelessly 
    copied from Elmer Solver manual.´and also check the formula there to understand
    what the coefficients mean.

    Parameters
    ----------
    k : int
        order of BD.

    Returns
    -------
    coefficients : np.ndarray shape (k+1)
        First one is multiplied with the forces and the stiffness matrix, the others on
        the right hand side with the mass matrix and the history of the function.
    """
    if k > 6:
        raise NotImplementedError("Not implemented yet for order higher than 6.")
    return np.array([[1,1],
                    [2/3,4/3,-1/3],
                    [6/11,18/11,-9/11,2/11],
                    [12/25,48/25,-36/25,16/25,-3/25],
                    [60/137,300/137,-300/137,200/137,-75/137,12/137],
                    [60/147,360/147,-450/147,400/147,-225/147,72/147,-10/147]]) 

def even_spaced_ternary(npoints):
    """
    Create even spaced points for a ternary phase system.

    Parameters
    ----------
    npoints : int
        number of points.

    Returns
    -------
    fracs : np.ndarray shape (k+1,3)
        phase fractions.
    """
    
    fracs = [] 
    for i,a in enumerate(np.linspace(0.0,1.0,npoints)):
        for b in np.linspace(0.0,1.0,npoints)[:(npoints-i)]:
            fracs.append([a,b,1-a-b])
    return np.array(fracs)

def parse_logfile_old(file):
    """
    Parse log file of the folding mechanism TO workflow. This is legacy and 
    will be deprecated soon.

    Parameters
    ----------
    file : str
        filename of the logfile.

    Returns
    -------
    params : dict
        contains some of the parameters like system size and shape, 
        optimizer etc..
    data : np.ndarray
        iteration history over objective function, volume constraint, change.

    """
    params = dict()
    with open(file,"r") as f:
        # 1st line
        params["optimizer"] = f.readline().strip().split(" ")[-1]
        # 2nd line
        line = f.readline().strip().split(" ")
        if len(line) == 4:
            nelx,nely = line[1::2] 
            params["nelx"] = int(nelx) 
            params["nely"] = int(nely)
        if len(line) == 6:
            nelx,nely,nelz = line[1::2]
            params["nelx"] = int(nelx) 
            params["nely"] = int(nely)
            params["nelz"] = int(nelz)
        # 3rd line
        line = f.readline().strip().split(" ")
        params["volfrac"] = float(line[1][:-1])
        params["rmin"] = float(line[3][:-1])
        params["penal"] = float(line[-1])
        # 4th line 
        params["filter"] = f.readline().strip().split(" ",2)[2]
        #
        lines = [line.replace(",","") for line in f]
        # last_line
        final = [float(i) for i in lines[-1].strip().split(" ")[2::2]]
        
    data = np.loadtxt(lines[:-1], delimiter=" ",
                      skiprows=0, usecols = [1,4,6,8]) 
    return params,data,final

def parse_logfile(file):
    """
    Parse log file of the compliance minimization TO workflow.

    Parameters
    ----------
    file : str
        filename of the logfile.

    Returns
    -------
    params : dict
        contains some of the parameters like system size and shape, 
        optimizer etc..
    data : np.ndarray
        iteration history over objective function, volume constraint, change.

    """
    
    params = dict()
    with open(file,"r") as f:
        # 1st line
        params["optimizer"] = f.readline().strip().split(" ")[-1]
        # 2nd line
        params["ndim"] = int(f.readline().strip().split(" ")[-1])
        # 3rd line
        line = f.readline().strip().split(" ")
        if len(line) == 4:
            nelx,nely = line[1::2] 
            params["nelx"] = int(nelx) 
            params["nely"] = int(nely)
        if len(line) == 6:
            nelx,nely,nelz = line[1::2]
            params["nelx"] = int(nelx) 
            params["nely"] = int(nely)
            params["nelz"] = int(nelz)
        # 4th line
        line = f.readline().strip().split(" ")
        params["volfrac"] = float(line[1][:])
        params["rmin"] = float(line[3][:])
        params["penal"] = float(line[-1])
        # 5th line 
        params["filter"] = f.readline().strip().split(" ",1)[1]
        # 6th line
        params["filter method"] = f.readline().strip().split(" ",1)[1]
        # 
        lines = [line.replace(",","") for line in f]
        # last_line
        #final = [float(i) for i in lines[-1].strip().split(" ")[2::2]]
        
    data = np.loadtxt(lines, delimiter=" ",
                      skiprows=0, usecols = [1,3,5,7]) 
    return params,data

def unique_sort(iM,jM,combine=False):
    """
    Sort first according to iM, then sort values of equal value iM according 
    to jM.

    Parameters
    ----------
    iM : np.ndarray, shape (n)
        first array.
    jM : np.ndarray, shape (n)
        second array.
    combine : scalar bool
        if True, stack both to to a column array of shape (n,2) 

    Returns
    -------
    iM : np.ndarray shape (n)
        if not combine returns sorted iM
    jM : np.ndarray shape (n)
        if not combine returns sorted jM
    M : np.ndarray shape (n,2)
        if combine returns column stack of sort iM and jM.

    """
    
    inds = np.lexsort((jM,iM))
    if combine:
        return np.column_stack((iM[inds],jM[inds])) 
    else:
        return iM[inds],jM[inds]

def map_eltoimg(quant,nelx,nely,**kwargs):
    """
    Map quantity located on elements on the usual regular grid to an image.

    Parameters
    ----------
    quant : np.ndarray, shape (n,nchannel)
        some quantity defined on each element (e. g. element density).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
        
    Returns
    -------
    img : np.ndarray, shape (nely,nelx,nchannel)
        quantity mapped to image.

    """
    #
    shape = (nely,nelx)+quant.shape[1:]
    return quant.reshape(shape,order="F")

def map_imgtoel(img,nelx,nely,**kwargs):
    """
    Map image of quantity back to 1D np.ndarray with correct (!) ordering.

    Parameters
    ----------
    img : np.ndarray, shape (nely,nelx)
        image of quantity (e. g. of element densities).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.

    Returns
    -------
    quant : np.ndarray, shape (n)
        quantity mapped back to 1D np.ndarray with hopefully correct ordering.

    """
    shape = tuple([nelx*nely])+img.shape[2:]
    return img.reshape(shape,order="F")

def map_eltovoxel(quant,nelx,nely,nelz,**kwargs):
    """
    Map quantity located on elements on the usual regular grid to a voxels.

    Parameters
    ----------
    quant : np.ndarray, shape (n,nchannel)
        some quantity defined on each element (e. g. element density).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    voxel : np.ndarray shape (nelz,nely,nelx,nchannel)
        quantity mapped to voxels.

    """
    #
    shape = (nelz,nelx,nely)+quant.shape[1:]
    return quant.reshape(shape).transpose((0,2,1)+tuple(range(3,len(shape))))

def map_voxeltoel(voxel,nelx,nely,nelz,**kwargs):
    """
    Map voxels of quantity back to on elements on the usual regular grid.

    Parameters
    ----------
    voxel : np.ndarray, shape (nelz,nely,nelx,nchannel)
        voxels of quantity (e. g. of element densities).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    quant : np.ndarray, shape (n,nchannel)
        quantity mapped back to 1D np.ndarray with hopefully correct ordering.

    """
    shape = tuple([nelx*nely*nelz])+voxel.shape[3:]
    voxel = voxel.transpose((0,2,1)+tuple(range(3,len(voxel.shape))))
    return voxel.reshape(shape)

def elid_to_coords(el,nelx,nely,nelz=None,**kwargs):
    """
    Map element ids to cartesian coordinates in the usual regular grid.

    Parameters
    ----------
    el : np.ndarray, shape (n)
        elment IDs.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    img : np.ndarray shape (nely,nelx)
        quantity mapped to image.

    """
    # find coordinates of each element/density
    if nelz is None:
        x,y = np.divmod(el,nely) # same as np.floor(el/nely),el%nely
        return x,y
    else:
        z,rest = np.divmod(el,nelx*nely)
        x,y = np.divmod(rest,nely)
        return x,y,z