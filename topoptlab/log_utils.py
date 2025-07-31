from typing import Callable
from os.path import isfile
from os import remove
import logging
from sys import platform

def init_logging(logfile: str) -> Callable:
    """
    Initialize the logging and returns a function for the specified logging.

    Parameters
    ----------
    logfile : str
        name of logfile.

    Returns
    -------
    logging_function : callable
        Returns the function used to write information to logfile.

    """
    if platform == "linux":
        # check if log file exists and if True delete
        if isfile(".".join([logfile,"log"])):
            remove(".".join([logfile,"log"])) 
        # check if any previous loggers exist and close them properly, 
        # otherwise you start writing the same information in a single huge 
        # file
        logger = logging.getLogger()
        if logger.hasHandlers():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()  
        #
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            handlers=[logging.FileHandler(".".join([logfile,"log"])),
                                      logging.StreamHandler()])
        return logging.info
    elif platform in ["win32"]:
        return WindowsLogging(logfile)
    else:
        return print
    return

class WindowsLogging:
    """
    Simple class to allow logging on Windows as there the logging module 
    continues running in the background even after the program has finished. 
    This can probably be avoided by calling python from the command line, but 
    to my experience nobody does this but runs it in some IDE.
    """
    
    def __init__(self,file: str) -> None:
        """
        Initiate logging by creating the logfile.

        Parameters
        ----------
        file : str
            name of logfile.

        Returns
        -------
        None.

        """
        
        self.file = file
        # creates empty file
        with open(self.file,"w") as f:
            pass
        return
    
    def __call__(self, msg: str) -> None:
        """
        Append message to logfile.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        with open(self.file,"a") as f:
            f.write(msg)
        print(msg)
        return