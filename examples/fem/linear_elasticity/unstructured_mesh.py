# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.geometry_parser import parse_cad_and_mesh,boxunion_meshing

if __name__ == "__main__":
    
    import sys
    if len(sys.argv)>1: 
        pass
    else:
        #äparse_cad_and_mesh(file="box.step",
        #               mesh_dim=3, 
        #               mesh_file="output.msh",
        #               show_gui=True)
        boxunion_meshing(file="test.step",
                         mesh_dim=3, 
                         mesh_file="output.msh",
                         show_gui=True)