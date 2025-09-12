import pytest

@pytest.mark.skip(reason="gmsh dependency not available in CI")

def test_parse_cad_and_mesh():
    
    from topoptlab.geometry_parser import parse_cad_and_mesh
    
    parse_cad_and_mesh(file="./tests/test_files/box.step",
                   mesh_dim=3, 
                   mesh_file="output.msh",
                   show_gui=False)
