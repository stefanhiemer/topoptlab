# SPDX-License-Identifier: GPL-3.0-or-later

from symfem import create_element, create_reference
from symfem.references import Reference
from symfem.finite_element import FiniteElement
from symfem.functionals import PointEvaluation


def node_ordering(element: FiniteElement,
                  convention: dict = {
    "interval": {"corners": ((-1,), (1,)),
                 "edges": ((0, 1),),
                 "faces": ()},
    "quadrilateral": {"corners": ((-1, -1), (1, -1), (1, 1), (-1, 1)),
                      # CPS8: 1-2, 2-3, 3-4, 4-1
                      "edges": ((0, 1), (1, 2), (2, 3), (3, 0)),
                      "faces": ((0, 1, 2, 3),)},
    "hexahedron": {"corners": ((-1, -1, -1), (1, -1, -1), (1, 1, -1),
                               (-1, 1, -1),
                               (-1, -1, 1), (1, -1, 1), (1, 1, 1),
                               (-1, 1, 1)),
                   # C3D20: bottom, top, vertical
                   "edges": ((0, 1), (1, 2), (2, 3), (3, 0),
                             (4, 5), (5, 6), (6, 7), (7, 4),
                             (0, 4), (1, 5), (2, 6), (3, 7)),
                   # ABAQUS faces S1-S6. Order of face nodes for C3D27 not
                   # yet verified against ABAQUS documentation.
                   "faces": ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)),
                   "volumes": ((0, 1, 2, 3, 4, 5, 6, 7),)}}) -> tuple[list, tuple]:
    """
    Determine node ordering and node coordinates for an element of symfem. 
    
    The nodes are sorted in the classical (ABAQUS-like) convention by nodes on 
    
        1. corners
        2. edges
        3. faces
        4. interior
    
    In each group, nodes are sorted by the number of their entity in the
    convention (see convention below) and within each entity by their exact
    position relative to the start corner of the entity, i. e. along the edge
    or lexicographically in the local coordinates of the face/volume (second
    direction slowest).

    The node coordinates are extracted from the point evaluations of the
    degrees of freedom of symfem, so everything is exact (sympy Rationals).

    Parameters
    ----------
    element : symfem.finite_element.FiniteElement
        element of symfem as created by symfem.create_element. Its reference
        cell must be "interval", "quadrilateral" or "hexahedron" and all its
        dofs must be point evaluations (nodal elements), otherwise
        NotImplementedError is raised.
    convention : dict
        node numbering conventions on [-1,1]^ndim. Keys are the cell names,
        values dicts with the entries

        - "corners": tuple of corner coordinates in the order of numbering,
        - "edges": tuple of edges given as tuples of corner indices,
        - "faces": tuple of faces given as tuples of corner indices,
        - "volumes" (optional): tuple of volumes given as tuples of corner
          indices.

        The order of the tuples is the order of the nodes on the respective
        entities, the order within each tuple defines the orientation of the
        entity: the first corner is the start corner, the second corner
        defines the first local direction and for faces the last corner the
        second local direction. Default is the classical (ABAQUS-like)
        convention: corners numbered counterclockwise starting at
        (-1,...,-1), first the bottom layer, then the top layer, edges and
        faces as in CPS8/C3D20 and the ABAQUS faces S1-S6. The order of the
        face nodes for C3D27 is not yet verified against the ABAQUS
        documentation.

    Returns
    -------
    nd_inds : list
        nd_inds[i] is the index of the basis function/dof of symfem that
        belongs to node i of topoptlab.
    coords : tuple
        coordinates of the nodes on the cell [-1,1]^ndim in topoptlab
        ordering.
    """
    reference = element.reference
    corners, entities, sf_to_corner = _cell_topology(reference=reference,
                                                     convention=convention)
    ndim = len(corners[0])
    #
    keys = []
    # only nodal elements: every dof must be the point evaluation of a
    # scalar basis function. Other functionals (integral moments, derivatives,
    # dot products, ...) also have a dof_point, so check the type.
    if not hasattr(element, "dofs"):
        raise NotImplementedError(f"{type(element).__name__} is not a "
                                  "Ciarlet element (no dofs).")
    invalid = sorted({type(dof).__name__ for dof in element.dofs \
                      if type(dof) is not PointEvaluation})
    if invalid:
        raise NotImplementedError(f"{type(element).__name__} has dofs that "
                                  "are not point evaluations: "
                                  f"{', '.join(invalid)}. Only nodal "
                                  "elements are supported.")
    for i, dof in enumerate(element.dofs):
        coord = tuple(2*c-1 for c in dof.dof_point())
        dim, dim_nr = dof.entity # dim of entity and which dim entity it belongs to
        # corners of the symfem entity
        sub = reference.sub_entity(dim, dim_nr)
        entity_corners = {sf_to_corner[reference.vertices.index(v)] \
                          for v in sub.vertices}
        # entity of convention
        k = [set(e) for e in entities[dim]].index(entity_corners)
        entity = entities[dim][k]
        # local coordinates relative to start corner along directions to the
        # neighbouring corners of the entity (edge: one, face: two, volume:
        # three)
        start = corners[entity[0]]
        if dim == 0:
            local = ()
        elif dim == 3:
            local = tuple(coord[j]-start[j] for j in range(ndim))
        else:
            # edges: second corner, faces: second and last corner
            ends = [entity[1]] if dim == 1 else [entity[1], entity[-1]]
            local = tuple(sum((coord[j]-start[j])*(corners[e][j]-start[j]) \
                              for j in range(ndim)) for e in ends)
        keys.append((dim, k, tuple(reversed(local)), i))
    nd_inds = [key[-1] for key in sorted(keys)]
    coords = tuple(tuple(2*c-1 for c in element.dofs[i].dof_point()) \
                   for i in nd_inds)
    return nd_inds, coords

def _cell_topology(reference: Reference,
                   convention: dict) -> tuple[tuple, dict, list]:
    """
    Topology of the reference cell of symfem in the given node numbering
    convention.

    Parameters
    ----------
    reference : symfem.references.Reference
        reference cell of symfem.
    convention : dict
        node numbering conventions. See node_ordering.

    Returns
    -------
    corners : tuple
        corner coordinates on [-1,1]^ndim in the order of the convention.
    entities : dict
        keys are the entity dimensions (0: corners, 1: edges, 2: faces,
        3: volumes), values tuples of entities given as tuples of corner
        indices in the order of the convention.
    sf_to_corner : list
        sf_to_corner[i] is the corner index in the convention of vertex i of
        the symfem reference cell.
    """
    if reference.name not in convention:
        raise NotImplementedError(f"cell {reference.name} not implemented.")
    convention = convention[reference.name]
    corners = convention["corners"]
    # entities of the convention by dimension
    entities = {0: tuple((i,) for i in range(len(corners))),
                1: convention["edges"],
                2: convention["faces"],
                3: convention.get("volumes", ())}
    # symfem vertices mapped from [0,1] to [-1,1] and to corner indices
    sf_to_corner = [corners.index(tuple(2*c-1 for c in v)) \
                    for v in reference.vertices]
    return corners, entities, sf_to_corner

def base_cell(ndim: int,
              element_type: str = "Lagrange",
              order: int = 1) -> tuple[tuple,list,Reference,list]:
    """
    Create the basic cell, location of nodes, the node indices, the
    reference cell and the basis functions. The nodes are ordered in the
    classical (ABAQUS-like) way, see node_ordering.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    vertices : tuple
        coordinates of all nodes on the cell [-1,1]^ndim in topoptlab
        ordering. The first 2**ndim nodes are the corners.
    nd_inds : list
        nd_inds[i] is the index of the basis function of symfem that belongs
        to node i of topoptlab.
    reference : symfem.references.Reference
        reference cell object.
    basis : list
        list of basis functions in topoptlab ordering.
    """
    if ndim == 1:
        cell_name = "interval"
    elif ndim == 2:
        cell_name = "quadrilateral"
    elif ndim == 3:
        cell_name = "hexahedron"
    else:
        raise ValueError("ndim must be 1, 2 or 3.")
    # create element
    element = create_element(cell_name, element_type, order=order)
    # find node ordering and associated coordinates
    nd_inds, vertices = node_ordering(element=element)
    # corners in the ordering of the symfem reference cell mapped to [-1,1]
    corners = tuple(tuple(2*c-1 for c in v) \
                    for v in element.reference.vertices)
    # create reference cell with these vertices
    reference = create_reference(cell_name, vertices=corners)
    # map the basis functions to the cell
    basis = element.map_to_cell(corners)
    # reorder basis function according to the current node ordering
    basis = [basis[i] for i in nd_inds]
    return vertices, nd_inds, reference, basis
