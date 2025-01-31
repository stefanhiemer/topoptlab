import numpy as np

from topoptlab.filters import assemble_matrix_filter,assemble_convolution_filter

if __name__ == "__main__":
    #
    rmin = 1.5
    nelx=5
    nely=2
    #
    dc = np.array([-705.65535777, -669.20970383, 
                   -412.529628, -447.82532496,
                   -232.57893823, -231.60167751, 
                   -107.68638974, -101.11611645,
                   -31.49861649, -126.63496201])
    
    H,Hs = assemble_matrix_filter(nelx,nely,rmin)
    h,hs = assemble_convolution_filter(nelx,nely,rmin)
    
    test = np.asarray(H*(dc[:,np.newaxis]/Hs))[:, 0]
    test1 = np.asarray((H*dc)[:,None]/Hs)[:, 0]
    
    print(test)
    print(test1)