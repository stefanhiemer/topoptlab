from topoptlab.geometry_parser import parse_cad_and_mesh

if __name__ == "__main__":
    
    import sys
    if len(sys.argv)>1: 
        pass
    else:
        parse_cad_and_mesh(file="box.step",
                       mesh_dim=3, 
                       mesh_file="output.msh",
                       show_gui=True)