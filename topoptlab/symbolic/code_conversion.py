# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple
from io import StringIO
import sys
from re import sub

from symfem.functions import MatrixFunction

def convert_to_code(matrix: MatrixFunction,
                    matrices: List = [],
                    vectors: List = [],
                    matrices_ele: List = [],
                    vectors_ele: List = [],
                    np_functions: List = ["cos","sin","tan","exp"],
                    npndarray: bool = False,
                    npcolumnstack: bool = True,
                    max_line_length: int = 200,
                    indent: int = 0) -> str:
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
    matrices_ele : list
        same as matrices, but are defined for each element independently. Only
        relevant for npcolumn_stack.
    vectors_ele : list
        same as vectors, but are defined for each element independently. Only
        relevant for npcolumn_stack.
    npndarray: bool
        if True, writes the output as numpy ndarray
    max_line_length : int
        counts number of length until first "]". If larger than the specified
        value, line breaks occur at every ",", otherwise at every "],".
    indent : int
        number of columns the first line will be shifted by once placed into
        its final location (e. g. after a "    return " prefix). Continuation
        lines are padded so they still align under the opening bracket at
        that location instead of column 0.

    Returns
    -------
    lines : str
        symfem output converted to code that can be copy pasted into a
        function.

    """
    #
    lines = symfemMatrixFunc_to_str(matrxfnc=matrix)
    #
    if npndarray:
        lines,delta = to_npndarray(lines=lines,
                                   max_line_length=max_line_length,
                                   indent=indent)
    elif npcolumnstack:
        lines,delta = to_npcolumn_stack(lines=lines,
                                        max_line_length=max_line_length,
                                        shape=matrix.shape,
                                        indent=indent)
    else:
        #
        first_line = lines.split("],",1)[0]
        # add line break after every comma
        if len(first_line) > max_line_length:
            lines = lines.replace(",",",\n"+" "*indent)
        # add line break after every "],"
        else:
            lines = lines.replace("],","],\n"+" "*indent)
    # add numpy prefix to functions
    for npfunc in np_functions:
        lines = lines.replace(npfunc,"np."+npfunc)
    # replace entries ala "c11" with corresponding array entries c[0,0]
    for matrix in matrices:
        if npndarray:
            lines = sub(matrix + r'(\d)(\d)',
                        lambda m: matrix +  f'[{int(m.group(1))-1},{int(m.group(2))-1}]',
                        lines)
        elif npcolumnstack:
            lines = sub(matrix + r'(\d)(\d)',
                        lambda m: matrix +  f'[:,{int(m.group(1))-1},{int(m.group(2))-1}]',
                        lines)
    # replace entries ala "c1" with corresponding array entries c[0]
    for vector in vectors:
        lines = sub(vector + r'(\d+)',
                    lambda m: vector + f'[{int(m.group(1))-1}]',
                    lines)
    if npcolumnstack:
        for vector in vectors_ele:
            lines = sub(vector + r'(\d+)',
                        lambda m: vector + f'[:,{int(m.group(1))-1}]',
                        lines)
        for matrix in matrices:
            lines = sub(matrix + r'(\d)(\d)',
                        lambda m: matrix +  f'[:,{int(m.group(1))-1},{int(m.group(2))-1}]',
                        lines)
    return lines

def to_npndarray(lines: List,
                 max_line_length: int,
                 indent: int = 0) -> Tuple[List,int]:
    """
    Convert the collected symfem string output to np.ndarray conform strings
    and formatting.

    Parameters
    ----------
    lines : str
        collected symfem output.
    max_line_length : int
        counts number of length until first "]". If larger than the specified
        value, line breaks occur at every ",", otherwise at every "],".
    indent : int
        number of columns the first line will be shifted by once placed into
        its final location. Continuation lines are padded so they still
        align under the opening bracket at that location instead of column 0.

    Returns
    -------
    lines : str
        converted lines.

    """
    #
    first_line = lines.split("],",1)[0]
    #
    delta = indent + len("np.array("+first_line) - len(first_line)
    # add np.array
    lines = "np.array(" + lines
    lines = lines[:-1] + ")"
    # add line break after every comma
    if len(first_line) > max_line_length:
        lines = lines.replace(",",",\n"+"".join([" "]*(delta+1)))
        lines = lines.replace(" [","[")
    # add line break after every "],"
    else:
        lines = lines.replace("],","],\n"+"".join([" "]*delta))
    return lines,delta

def to_npcolumn_stack(lines: List,
                      max_line_length: int,
                      shape: Tuple,
                      indent: int = 0) -> Tuple[List,int]:
    """
    Convert the collected symfem string output to np.column_stack conform
    strings and formatting.

    Parameters
    ----------
    lines : str
        collected symfem output.
    max_line_length : int
        counts number of length until first "]". If larger than the specified
        value, line breaks occur at every ",", otherwise at every "],".
    shape : tuple
        shape of elemental matrix.
    indent : int
        number of columns the first line will be shifted by once placed into
        its final location. Continuation lines are padded so they still
        align under the opening bracket at that location instead of column 0.

    Returns
    -------
    lines : str
        converted lines.

    """
    #
    first_line = lines.split("],",1)[0]
    #
    delta = indent + len("np.column_stack("+first_line) - len(first_line)
    # add np.column_stack 
    lines = "np.column_stack((" + lines
    lines = lines[:-1] + "))"
    # add the necessary reshape
    lines = lines + ".reshape(-1,"+",".join([str(_) for _ in shape])+")"
    # add line break after every comma
    if len(first_line) > max_line_length:
        lines = lines.replace(",",",\n"+"".join([" "]*(delta+1)))
        lines = lines.replace(" [","[")
    # add line break after every "],"
    else:
        lines = lines.replace("],","],\n"+"".join([" "]*delta))
    # eliminate brackets for lists
    lines = lines.replace("[","")
    lines = lines.replace("]","")
    return lines,delta

def wrap_function(name: str,
                  signature: str,
                  matrix: MatrixFunction,
                  matrices: List = [],
                  vectors: List = [],
                  matrices_ele: List = [],
                  vectors_ele: List = [],
                  np_functions: List = ["cos","sin","tan","exp"],
                  npndarray: bool = False,
                  npcolumnstack: bool = True,
                  max_line_length: int = 200,
                  indent: str = "    ") -> str:
    """
    Wrap the code generated by convert_to_code into a full function
    definition without a docstring. The signature is used verbatim, so
    typing, defaults and **kwargs must already be included in it. Continuation
    lines of the returned array expression are aligned exactly as they would
    be if the raw convert_to_code output had been pasted directly after the
    "return " keyword.

    Parameters
    ----------
    name : str
        name of the function.
    signature : str
        function signature exactly as it should appear between the
        parentheses of the "def" statement, e. g.
        "p: float = 1., l: np.ndarray = np.array([1.,1.]), **kwargs: Any".
    matrix : symfem.functions.MatrixFunction
        symfem output.
    matrices : list
        see convert_to_code.
    vectors : list
        see convert_to_code.
    matrices_ele : list
        see convert_to_code.
    vectors_ele : list
        see convert_to_code.
    np_functions : list
        see convert_to_code.
    npndarray : bool
        see convert_to_code.
    npcolumnstack : bool
        see convert_to_code.
    max_line_length : int
        see convert_to_code.
    indent : str
        indentation used for the function body.

    Returns
    -------
    function_str : str
        full function definition (no docstring) that can be copy pasted into
        a module.

    """
    #
    prefix = indent + "return "
    body = convert_to_code(matrix=matrix,
                           matrices=matrices,
                           vectors=vectors,
                           matrices_ele=matrices_ele,
                           vectors_ele=vectors_ele,
                           np_functions=np_functions,
                           npndarray=npndarray,
                           npcolumnstack=npcolumnstack,
                           max_line_length=max_line_length,
                           indent=len(prefix))
    return f"def {name}({signature}):\n{prefix}{body}\n"

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