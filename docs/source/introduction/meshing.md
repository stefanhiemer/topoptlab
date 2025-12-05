# Meshing in Topology Optimization
In topology optimization (TO) usually three different meshes are used: 
stuctured, unstructured and body-fitted meshes. 

(fig-reg-vs-irreg)=
**Figure:** Structured (left) and unstructured mesh (right).

 ![](/_static/reg-mesh.png){width=300px} ![](/_static/irreg-mesh.png){width=300px}

The common choice in TO, a stuctured mesh is a structured grid, usually 
Cartesian, where the entire design domain is discretized into uniform elements. 
Besides their implementation simplicity, these meshes open the door to 
geometric multigrid solvers (GMG) which can increase numerical efficiency of 
the underlying finite element problem by orders of magnitude and can also make 
use of details specific to TO to increase efficiency further {cite}`amir2014multigrid`.

Unstructured meshes are built from elements that do not follow a regular grid 
pattern. Instead of repeating rectangles or cubes, the elements can have 
varying shapes and sizes—most often triangles in two dimensions or tetrahedra 
in three dimensions—that are placed in a flexible way to conform to arbitrary 
geometries. In TO, most the time irregular mehes offer no advantage in most 
cases and further complicate implementation of methods. Why no advantages? 
Typically irregular meshes are used in FE to accomodate sections of the 
geometry that are either very complex or where a high accuracy of the numerical 
solution is needed. Complex geometries do not exist in TO a-priori as finding 
the final design is why one does TO in the first place. Physical accuracy is 
needed in TO everywhere - although some shortcuts are possible here, but more 
at another time - as otherwise the constrained optimizer will exploit poor 
numerics to create designs that perform well for the given numerical parameters b
ut will perform poor with accurate numerics and thus also in experiments. 

A body-fitted mesh is created by discretizing the (final) TO design with an 
irregular mesh to offer a better geometrical and physical description which may
be needed as sanity check on the final TO design or during the TO iteration if 
accurate description is detrimental (e. g. contact problems) or if the specific
TO method (e. g. level set based TO with cut-element methdo) easily allows for 
the creation of the body-fitting on-the-fly.

## Regular Mesh used in topoptlab

In `topoptlab`, in most cases we use a mesh as displayed in this subsection and 
we assume our elements to be in the reference domain spanning [-1,1] in each 
spatial dimesion. In 2d the local degrees of freedom in the elements are 
arranged counter-clockwise starting from the node located at (-1,-1). In 3d the 
same method is used where we start from node (-1,-1,-1), number all nodes with
$z=-1$ and then repeat the same for the nodes $z=1$. This is the same approach
as in the famous 88 line code{cite}`andreassen2011efficient` and its updated version {cite}`ferrari2020new`
For scalar fields (e. g. temperature) the global degrees of freedom are 
enumerated as follows (with elements as roman numbers) in 2D

```{image} /_static/meshnumbering-scalar-2d.png
:align: center
:alt: Mesh numbering
:width: 400px
```

and in 3D 

```{image} /_static/meshnumbering-scalar-3d.png
:align: center
:alt: Mesh numbering
:width: 600px
```

Vector fields in 2D are enumerated as

```{image} /_static/meshnumbering-vector-2d.png
:align: center
:alt: Mesh numbering
:width: 400px
```

and in 3D as

```{image} /_static/meshnumbering-vector-3d.png
:align: center
:alt: Mesh numbering
:width: 600px
```
 
To better view these numberings, you also directly plot them via 
[`topoptlab_mesh.py`](https://github.com/stefanhiemer/topoptlab/blob/main/docs/source/_static/topoptlab_mesh.py).