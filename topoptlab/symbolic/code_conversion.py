# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List
from io import StringIO
import sys
from re import sub

from symfem.functions import MatrixFunction

def convert_to_code(matrix: MatrixFunction,
                    matrices: List = [], vectors: List = [],
                    np_functions: List = ["cos","sin","tan"],
                    npndarray: bool = True,
                    max_line_length: int = 200) -> str:
    """
    Convert the printed expression by symfem to strings that can be
    converted to code.

    Parameters
    ----------
    matrix : symfem.functions.MatrixFunction
        symfem output.
    matrices : list
        list of strs for the tensor indices to be converted to array indices.
        E. g. the tensor "c" appears in the equation, the current element
        derivation routines will return function that contain the elements of
        this tensor in the format c11,c12,etc. This function converts these
        entries to c[0,0],c[0,1],etc.
    vectors : list
        list of strs with same logic as matrices, but instead c1,c2,etc. are
        converted to c[0],c[1],etc.
    npndarray: bool
        if True, writes the output as numpy ndarray
    max_line_length : int
        counts number of length until first "]". If larger than the specified
        value, line breaks occur at every ",", otherwise at every "],".

    Returns
    -------
    lines : str
        symfem output converted to code that can be copy pasted into a
        function.

    """
    #
    lines = symfemMatrixFunc_to_str(matrxfnc=matrix)
    #
    first_line = lines.split("],",1)[0]
    #
    if npndarray:
        #
        delta = len("np.array("+first_line) - len(first_line)
        # add np.array
        lines = "np.array(" + lines
        lines = lines[:-1] + ")"
        #
        # add line break after every comma
        if len(first_line) > max_line_length:
            lines = lines.replace(",",",\n"+"".join([" "]*(delta+1)))
            lines = lines.replace(" [","[")
        # add line break after every "],"
        else:
            lines = lines.replace("],","],\n"+"".join([" "]*delta))
    else:
        # add line break after every comma
        if len(first_line) > max_line_length:
            lines = lines.replace(",",",\n")
        # add line break after every "],"
        else:
            lines = lines.replace("],","],\n")
    # add numpy prefix to functions
    for npfunc in np_functions:
        lines = lines.replace(npfunc,"np."+npfunc)
    # replace entries ala "c11" with corresponding array entries c[0,0]
    for matrix in matrices:
        lines = sub(matrix + r'(\d)(\d)',
              lambda m: matrix +  f'[{int(m.group(1))-1},{int(m.group(2))-1}]',
              lines)
    for vector in vectors:
        # replace entries ala "c1" with corresponding array entries c[0]
        lines = sub(vector + r'(\d)',
                    lambda m: vector + f'[{int(m.group(1))-1}]',
                    lines)
    return lines


def symfemMatrixFunc_to_str(matrxfnc: MatrixFunction) -> str:
    """
    Convert symfem MatrixFunction to str via print() and capture this.

    Parameters
    ----------
    matrxfnc : symfem.functions.MatrixFunction
        matrix to convert to str.

    Returns
    -------
    lines : str
        symfem MatrixFunction converted to str via print() and captured.

    """
    # convert symfem.MatrixFunction to list to better print it
    ls = []
    for i in range(matrxfnc.shape[0]):
        ls.append([])
        for j in range(matrxfnc.shape[1]):
            ls[-1].append(matrxfnc[i,j])
    # create a StringIO object to capture print output
    stringio_capturer = StringIO()
    # redirect stdout to the StringIO object
    sys.stdout = stringio_capturer
    # feed the matrix into the capturer
    print(ls)
    # reset stdout back to normal
    sys.stdout = sys.__stdout__
    # convert printed output to string
    lines = stringio_capturer.getvalue()
    stringio_capturer.close()
    return lines