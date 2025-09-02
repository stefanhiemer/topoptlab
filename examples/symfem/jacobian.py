from topoptlab.symbolic.parametric_map import jacobian

if __name__ == "__main__":


    #
    J, Jinv, Jdet = jacobian(ndim = 2,debug=True)
    print("Jacobian: ",J)
    print("inverse of Jacobian: ", Jinv) 
    print("determinant of Jacobian: ",Jdet)
    #print(convert_to_code(jacobian(ndim = 2)))
