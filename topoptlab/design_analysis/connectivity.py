# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from scipy.sparse import coo_array, csr_array, diags_array
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

def boundary_elements(adjacency: csr_array,
                      ndim: int) -> np.ndarray:
    """
    Find the elements with at least one face on the boundary of the mesh, so
    elements that have fewer face neighbors than faces. For meshes with
    periodic boundary conditions, the faces on periodic boundaries are
    interior faces and therefore not found. Valid for linear Lagrangian
    elements (intervals, quadrilaterals, hexahedra).

    Parameters
    ----------
    adjacency : scipy.sparse.csr_array, shape (n_el,n_el)
        face adjacency graph, e. g. from topoptlab.utils.grid_adjacency or
        topoptlab.utils.element_adjacency.
    ndim : int
        number of spatial dimensions.

    Returns
    -------
    boundary : np.ndarray, shape (n_el)
        True for elements on the boundary.

    """
    return (adjacency != 0).sum(axis=1) < 2*ndim

def phase_components(x: np.ndarray,
                     adjacency: csr_array,
                     phase: str = "solid",
                     threshold: float = 0.5) -> tuple[int, np.ndarray]:
    """
    Label the connected components of one phase of a thresholded design.

    Parameters
    ----------
    x : np.ndarray, shape (n_el) or (n_el,1)
        element densities.
    adjacency : scipy.sparse.csr_array, shape (n_el,n_el)
        element adjacency graph.
    phase : str
        either "solid" (x > threshold) or "void" (x <= threshold).
    threshold : float
        threshold to binarize the design.

    Returns
    -------
    n_comp : int
        number of connected components of the phase.
    labels : np.ndarray, shape (n_el)
        component index of each element and -1 for elements of the other
        phase.

    """
    #
    inphase = _phase_mask(x=x, phase=phase, threshold=threshold)
    inds = np.flatnonzero(inphase)
    #
    labels = np.full(inphase.shape[0], -1)
    n_comp, labels[inds] = connected_components(adjacency[inds][:,inds],
                                                directed=False)
    return n_comp, labels

def void_islands(x: np.ndarray,
                 adjacency: csr_array,
                 boundary: np.ndarray,
                 threshold: float = 0.5) -> tuple[int, np.ndarray]:
    """
    Find enclosed voids (void islands), so void components that do not touch
    the given boundary elements. With all boundary elements of the mesh
    (see boundary_elements) this detects voids that are not connected to the
    outside, e. g. for powder removal in additive manufacturing. A design
    without void islands and without solid islands is called simply
    connected. See Sec. 2.1 in

    Cool, Vanessa, Niels Aage, and Ole Sigmund. "A practical review on
    promoting connectivity in topology optimization." arXiv preprint
    arXiv:2501.09402 (2025).

    Parameters
    ----------
    x : np.ndarray, shape (n_el) or (n_el,1)
        element densities.
    adjacency : scipy.sparse.csr_array, shape (n_el,n_el)
        element adjacency graph.
    boundary : np.ndarray, shape (n_el)
        boolean mask of elements through which void may escape.
    threshold : float
        threshold to binarize the design.

    Returns
    -------
    n_islands : int
        number of void islands.
    islands : np.ndarray, shape (n_el)
        True for elements belonging to a void island.

    """
    n_comp, labels = phase_components(x=x, adjacency=adjacency,
                                      phase="void", threshold=threshold)
    return _islands(n_comp=n_comp, labels=labels, anchor=boundary)

def solid_islands(x: np.ndarray,
                  adjacency: csr_array,
                  anchor: None | np.ndarray = None,
                  threshold: float = 0.5) -> tuple[int, np.ndarray]:
    """
    Find solid islands, so solid components that are not connected to any
    of the anchor elements (e. g. supports or passive solid regions). See
    Sec. 2.1 in

    Cool, Vanessa, Niels Aage, and Ole Sigmund. "A practical review on
    promoting connectivity in topology optimization." arXiv preprint
    arXiv:2501.09402 (2025).

    Parameters
    ----------
    x : np.ndarray, shape (n_el) or (n_el,1)
        element densities.
    adjacency : scipy.sparse.csr_array, shape (n_el,n_el)
        element adjacency graph.
    anchor : None or np.ndarray, shape (n_el)
        boolean mask of elements that solid has to be connected to. If None,
        the largest solid component is the structure and all others are
        islands.
    threshold : float
        threshold to binarize the design.

    Returns
    -------
    n_islands : int
        number of solid islands.
    islands : np.ndarray, shape (n_el)
        True for elements belonging to a solid island.

    """
    n_comp, labels = phase_components(x=x, adjacency=adjacency,
                                      phase="solid", threshold=threshold)
    if anchor is None:
        anchor = np.zeros(labels.shape[0], dtype=bool)
        if n_comp > 0:
            anchor[labels == np.bincount(labels[labels >= 0]).argmax()] = True
    return _islands(n_comp=n_comp, labels=labels, anchor=anchor)

def graph_laplacian(adjacency: csr_array,
                    w: np.ndarray) -> csr_array:
    """
    Weighted graph Laplacian

        L_ij = -w_ij for i != j
        L_ii = sum_j w_ij

    Parameters
    ----------
    adjacency : scipy.sparse.csr_array, shape (n_el,n_el)
        element adjacency graph.
    w : np.ndarray, shape (n_edges)
        weights of the upper triangular entries of the adjacency graph in the
        order of scipy.sparse.triu(adjacency).tocoo().

    Returns
    -------
    L : scipy.sparse.csr_array, shape (n_el,n_el)
        graph Laplacian.

    """
    #
    i, j = _edges(adjacency)
    n = adjacency.shape[0]
    W = coo_array((np.concatenate((w, w)),
                   (np.concatenate((i, j)), np.concatenate((j, i)))),
                  shape=(n, n)).tocsr()
    return (diags_array(W.sum(axis=1)) - W).tocsr()

def spectral_connectivity(x: np.ndarray,
                          adjacency: csr_array,
                          fixed: np.ndarray,
                          phase: str = "solid",
                          q: float = 1.,
                          w_min: float = 1e-3,
                          m_min: float = 1e-6,
                          k: int = 5,
                          return_eigvecs: bool = False
                          ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Spectral graph connectivity measure in the spirit of

    Donoso, Alberto, Enrique Aranda, and David Ruiz. "A new approach based on
    spectral graph theory to avoiding enclosed holes in topology
    optimization." Computer Methods in Applied Mechanics and Engineering 393
    (2022): 114769.

    Donoso, Alberto, Enrique Aranda, and David Ruiz. "A general method based
    on the Dirichlet-Laplacian problem for connectivity in topology
    optimization." Structural and Multidisciplinary Optimization 67.7
    (2024): 132.

    Each element is a node of a weighted graph and the eigenvalue problem

        L(rho) phi = lambda M(rho) phi,   phi = 0 on fixed elements

    is solved with the edge weights (Eq. 15 of the first reference)

        w_ij = (rho_i rho_j)^q (1-w_min) + w_min

    and the diagonal mass M_ii = rho_i (1-m_min) + m_min. rho is x for the
    solid phase and 1-x for the void phase. Every component of the phase that
    is not connected to the fixed elements is only attached through the weak
    weights w_min and yields an eigenvalue of order w_min, while components
    connected to the fixed elements yield eigenvalues of order 1/L^2 with L
    the graph distance to the fixed elements. The number of eigenvalues far
    below the others therefore counts the islands. Elements of the other
    phase are coupled by w_min and carry the mass m_min, so they create
    spurious modes with eigenvalues of order w_min/m_min. m_min must
    therefore be much smaller than w_min. Typical choices of fixed
    elements: the mesh boundary for void islands (enclosed voids) or the
    supports for solid islands.

    Parameters
    ----------
    x : np.ndarray, shape (n_el) or (n_el,1)
        element densities.
    adjacency : scipy.sparse.csr_array, shape (n_el,n_el)
        element adjacency graph.
    fixed : np.ndarray, shape (n_el)
        boolean mask of elements with homogeneous Dirichlet condition.
    phase : str
        either "solid" or "void".
    q : float
        penalization exponent of the edge weights.
    w_min : float
        minimum edge weight to keep the problem well posed.
    m_min : float
        minimum mass. Must be much smaller than w_min.
    k : int
        number of smallest eigenvalues returned.
    return_eigvecs : bool
        if True, also return the eigenvectors.

    Returns
    -------
    lam : np.ndarray, shape (k)
        smallest eigenvalues in ascending order.
    phi : np.ndarray, shape (n_el,k)
        eigenvectors normalized w.r.t. M with zeros on the fixed elements.
        Only returned if return_eigvecs is True.

    """
    #
    rho = _phase_density(x=x, phase=phase)
    # graph Laplacian and mass
    i, j = _edges(adjacency)
    w = (rho[i]*rho[j])**q * (1-w_min) + w_min
    L = graph_laplacian(adjacency=adjacency, w=w)
    m = rho*(1-m_min) + m_min
    # apply Dirichlet conditions
    free = np.flatnonzero(~fixed)
    L = L[free][:,free]
    M = diags_array(m[free]).tocsr()
    #
    k = min(k, free.shape[0]-1)
    lam, phi_free = eigsh(L, k=k, M=M, sigma=0., which="LM")
    order = np.argsort(lam)
    lam, phi_free = lam[order], phi_free[:,order]
    if not return_eigvecs:
        return lam
    phi = np.zeros((rho.shape[0], k))
    phi[free] = phi_free
    return lam, phi

def _phase_mask(x: np.ndarray,
                phase: str,
                threshold: float) -> np.ndarray:
    """
    Boolean mask of the elements belonging to one phase of the thresholded
    design.

    Parameters
    ----------
    x : np.ndarray, shape (n_el) or (n_el,1)
        element densities.
    phase : str
        either "solid" (x > threshold) or "void" (x <= threshold).
    threshold : float
        threshold to binarize the design.

    Returns
    -------
    mask : np.ndarray, shape (n_el)
        True for elements belonging to the phase.

    """
    x = np.asarray(x).reshape(-1)
    if phase == "solid":
        return x > threshold
    elif phase == "void":
        return x <= threshold
    else:
        raise ValueError(f"unknown phase: {phase}")

def _phase_density(x: np.ndarray,
                   phase: str) -> np.ndarray:
    """
    Density of one phase, so x for the solid phase and 1-x for the void
    phase.

    Parameters
    ----------
    x : np.ndarray, shape (n_el) or (n_el,1)
        element densities.
    phase : str
        either "solid" or "void".

    Returns
    -------
    rho : np.ndarray, shape (n_el)
        density of the phase.

    """
    x = np.asarray(x).reshape(-1)
    if phase == "solid":
        return x
    elif phase == "void":
        return 1-x
    else:
        raise ValueError(f"unknown phase: {phase}")

def _edges(adjacency: csr_array) -> tuple[np.ndarray, np.ndarray]:
    """
    Edges of the adjacency graph, each edge counted once via its upper
    triangular entry (i<j). The order is the one of scipy.sparse.coo_array
    and defines the order of the edge weights in graph_laplacian.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_array, shape (n_el,n_el)
        symmetric element adjacency graph.

    Returns
    -------
    i : np.ndarray, shape (n_edges)
        first element of each edge.
    j : np.ndarray, shape (n_edges)
        second element of each edge.

    """
    A = adjacency.tocoo()
    mask = A.row < A.col
    return A.row[mask], A.col[mask]

def _islands(n_comp: int,
             labels: np.ndarray,
             anchor: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Find the components that contain no anchor element (islands).

    Parameters
    ----------
    n_comp : int
        number of components.
    labels : np.ndarray, shape (n_el)
        component index of each element and -1 for elements not belonging to
        any component (e. g. other phase).
    anchor : np.ndarray, shape (n_el)
        boolean mask of anchor elements.

    Returns
    -------
    n_islands : int
        number of islands.
    islands : np.ndarray, shape (n_el)
        True for elements belonging to an island.

    """
    anchored = np.unique(labels[anchor & (labels >= 0)])
    islands = np.setdiff1d(np.arange(n_comp), anchored)
    return islands.shape[0], np.isin(labels, islands)
