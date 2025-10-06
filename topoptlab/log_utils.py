# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Dict,Tuple
from abc import ABC, abstractmethod
from os.path import isfile
from os import remove
from sys import platform
import re

import numpy as np

def _noop(msg: str) -> None:
    """
    Ignore message and do not print to a logfile or the screen.

    Parameters
    ----------
    msg : str
        message.

    Returns
    -------
    None.

    """
    pass

class BaseLogger(ABC):
    """
    Base class for Loggers and show general structure.
    """
    
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize Logger by creating log file, etc.
        
        Returns
        -------
        None

        """
        raise NotImplementedError()
    
    @abstractmethod
    def __call__(self) -> None:
        """
        Append message to logfile bypassing vebosity. Usually a bad idea.

        Parameters
        ----------

        Returns
        -------
        None.

        """
        raise NotImplementedError()
    
    @abstractmethod
    def _write(self) -> None:
        """
        Append message to logfile.

        Parameters
        ----------

        Returns
        -------
        None.

        """
        raise NotImplementedError()
    
    @abstractmethod
    def debug(self):
        """
        Write debugging information.
        
        Returns
        -------
        None

        """
        raise NotImplementedError
    
    @abstractmethod
    def perf(self):
        """
        Write performance information.
        
        Returns
        -------
        None

        """
        raise NotImplementedError()
    
    @abstractmethod
    def info(self):
        """
        Write general simulation information.
        
        Returns
        -------
        None

        """
        raise NotImplementedError()
    
    @abstractmethod
    def warning(self):
        """
        Write warnings.
        
        Returns
        -------
        None

        """
        raise NotImplementedError()
    
    @abstractmethod
    def error(self):
        """
        Write error information. These should be "expected" errors.
        
        Returns
        -------
        None

        """
        raise NotImplementedError()
    
    @abstractmethod
    def critical(self):
        """
        Write information due to critical failure. These are "unexpected" 
        errors.
        
        Returns
        -------
        None

        """
        raise NotImplementedError()
        
class SimpleLogger(BaseLogger):
    """
    Simple class to allow logging like the logging module, but is much easier
    to handle. Originally this was written because in IDEs as there logging 
    module continues running in the background even after the program has 
    finished causing errors to occur due to re-initialization and such.
    """
    
    def __init__(self,
                 file: str,
                 verbosity: int = 20) -> None:
        """
        Initiate logging by creating the logfile and setting up logging functions.

        Parameters
        ----------
        file : str
            name of logfile. Automatically adds '.log' to the filename if not 
            done already.
        verbosity : int 
            verbosity level following the conventions of the logging module with 
            a few extra levels. Please be aware of the slightly strange logic of 
            the logging module: counterintuitively lower numerical values 
            sned more detailed (less severe) messagese. e. g. verbosity=10 
            activates all output while verbosity=20 omits performance and 
            debug messages.
                
                - 10 : DEBUG        detailed iteration or solver info
                - 15 : PERFORMANCE  timing or convergence summaries
                - 20 : INFO         high-level simulation information
                - 30 : WARNING      non-fatal issues
                - 40 : ERROR        severe problems
                - 50 : CRITICAL     critical failures

        Returns
        -------
        None.

        """
        #
        if file.endswith(".log"):
            self.file = file
        else: 
            self.file = f"{file}.log"
        # creates empty file
        with open(self.file,"w") as f:
            pass
        # overwrite logging functions according to verbosity level
        if verbosity > 10: 
            self.debug = _noop
        #
        if verbosity > 15:
            self.perf = _noop
        #
        if verbosity > 20:
            self.info = _noop
        #
        if verbosity > 30:
            self.warning = _noop
        #
        if verbosity > 40:
            self.error = _noop
        #
        if verbosity > 50:
            self.critical = _noop
        return
    
    def __call__(self, msg: str) -> None:
        """
        Append message to logfile bypassing vebosity. Usually a bad idea.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        return self._write(msg=msg)
    
    def _write(self, msg: str) -> None:
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
        # automatic line break
        if not msg.endswith("\n"):
            msg += "\n"
        with open(self.file, "a") as f:
            f.write(msg)
        print(msg, end="")
        return
    
    def debug(self, msg: str) -> None:
        """
        Append debugging information to logfile.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        return self._write(msg= "[DEBUG] "+ msg)
    
    def perf(self, msg: str) -> None:
        """
        Append performance information to logfile.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        return self._write(msg= "[PERF] "+ msg)
    
    def info(self, msg: str) -> None:
        """
        Append simulation information to logfile.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        return self._write(msg)
    
    def warning(self, msg: str) -> None:
        """
        Append warnings to logfile.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        return self._write("[WARNING] "+msg)
    
    def error(self, msg: str) -> None:
        """
        Append error messages to logfile. These errors are "expected" and 
        should stem from invalid combinations or usual things like convergence
        issues.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        return self._write("[ERROR] "+msg)
    
    def critical(self, msg: str) -> None:
        """
        Append critical error messages to logfile. These errors are unexpected
        and mean that either you have entered some nonsense which naturally 
        will lead to strange errors or something that we overlooked.

        Parameters
        ----------
        msg : str
            message.

        Returns
        -------
        None.

        """
        return self._write("[CRITICAL] "+msg)
    
class EmptyLogger(BaseLogger):
    """
    Imitates SimpleLogger in terms of the available methods, but never does 
    anything. This is just a placeholder class.
    """
    
    def __init__(self,
                 **kwargs: Any) -> None:
        """
        Initiate Logger, but does nothing.

        Parameters
        ----------
        None

        Returns
        -------
        None.

        """
        return
    
    def _write(self, *args, **kwargs: Any) -> None: pass
    def __call__(self, *args, **kwargs: Any) -> None: pass
    def debug(self, *args,**kwargs: Any) -> None: pass
    def perf(self, *args, **kwargs: Any) -> None: pass
    def info(self, *args, **kwargs: Any) -> None: pass
    def warning(self, *args, **kwargs: Any) -> None: pass
    def error(self, *args, **kwargs: Any) -> None: pass
    def critical(self, *args, **kwargs: Any) -> None: pass

def init_logging(logfile: str, 
                 verbosity: int = 20) -> SimpleLogger:
    """
    Initialize the logging and returns a function for the specified logging. 
    This function is right now purely legacy as we have abandoned the native 
    Python logging package. Might be revived later. 

    Parameters
    ----------
    logfile : str
        name of logfile.
    verbosity : int
        verbosity level following the conventions of the logging module with 
        a few extra levels. Please be aware of the slightly strange logic of 
        the logging module: counterintuitively lower numerical values 
        sned more detailed (less severe) messages. e. g. verbosity=10 
        activates all output while verbosity=20 omits performance and 
        debug messages.
            
            - 10 : DEBUG        detailed iteration or solver info
            - 15 : PERFORMANCE  timing or convergence summaries
            - 20 : INFO         high-level simulation information
            - 30 : WARNING      non-fatal issues
            - 40 : ERROR        severe problems
            - 50 : CRITICAL     critical failures

    Returns
    -------
    logger : SimpleLogger
        Returns the Logger used to write information to logfile.
    

    """
    # simple logging functionalities.
    if platform in ["win32","darwin","linux"]:
        return SimpleLogger(logfile, verbosity=verbosity)
    # python official logging module, but discourages custom verbosity levels,
    # which I however need.
    elif platform == []:
        #
        import logging
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
        return logger
    
    else:
        raise ValueError("Your platform is unknown and kills the logging. ",
                         "Check the return value of sys.platform and add it ",
                         "to the list in line 43 as quickfix. Otherwise ",
                         "contact the maintainers.")
    return

def parse_simple_logfile(path: str) -> Dict[str, Any]:
    """
    Parse a log file written by SimpleLogger into header information,
    iteration data, and tagged messages such as [PERF], [DEBUG], etc.
      
     Parameters
     ----------
     path : str
         path of logfile.

     Returns
     -------
     log_data : dict
         dictionary of information of logfile.
         
    Notes
    -----
    The parser stops collecting header information once it encounters  an 
    iteration line ("it.: ...") or any line beginning with a log prefix such as
    [PERF], [DEBUG], etc.
    """
    
    #
    log_data = {}
    # regex pattern of tagged messages
    tag_pattern = re.compile(r"^\[([A-Z]+)\]\s*(.*)")
    # pattern for finding key-value pairs in iteration information
    kv_pattern = re.compile(r"([A-Za-z_]+\.?):\s*([-+]?\d*\.\d+|\d+|[A-Za-z_./-]+)")
    # read logfile
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # extract lines containing the header
    header_lines = []
    for i, line in enumerate(lines):
        if line.startswith("it.:") or tag_pattern.match(line):
            start_idx = i
            break
        header_lines.append(line)
    else:
        start_idx = len(lines)
    
    # parse key–value pairs from header
    header_dict: Dict[str, Any] = dict()
    for line in header_lines:
        # detect elements line and extract nelx, nely, (nelz)
        if line.lower().startswith("elements:"):
            m = re.search(r"elements:\s*(\d+)\s*x\s*(\d+)(?:\s*x\s*(\d+))?", line)
            if m:
                params = ["nelx", "nely", "nelz"]
                for i, val in enumerate(m.groups()):
                    if val is not None:
                        header_dict[params[i]] = int(val)
            continue
        #
        if line.lower().startswith("filter method:"):
            value = line.split(":", 1)[-1].strip()
            header_dict["filter method"] = value
            continue
        #
        kvs = kv_pattern.findall(line)
        if kvs:
            for k, v in kvs:
                key = k.strip(":")
                try:
                    header_dict[key] = float(v)
                except ValueError:
                    header_dict[key] = v
        else:
            header_dict[f"line_{len(header_dict)}"] = line
    # add header information
    log_data["header"] = header_dict
    # extract iteration data and tagged information
    tagged = dict()
    data_list = []
    for line in lines[start_idx:]:
        # design iteration information
        if line.startswith("it.:"):
            kvs = kv_pattern.findall(line)
            entry: Dict[str, Any] = dict()
            for k, v in kvs:
                key = k.strip(".: ")
                try:
                    entry[key] = float(v)
                except ValueError:
                    entry[key] = v
            data_list.append(entry)
        # find tagged information
        tag_match = tag_pattern.match(line)
        if tag_match:
            tag, msg = tag_match.groups()
            tag = tag.lower()
            if tag not in tagged:
                tagged[tag] = []
            tagged[tag].append(msg)
            continue
    # package all in one array
    if len(data_list) > 0:
        # collect all unique keys
        all_keys = set().union(*(d.keys() for d in data_list))
        data = {}
        for key in all_keys:
            try:
                data[key] = np.array([float(d.get(key, np.nan)) for d in data_list])
            except ValueError:
                # fallback if a key is non-numeric
                data[key] = np.array([d.get(key, "") for d in data_list])
    else:
        data = {}
    #
    #log_data["tagged"] = tagged
    log_data["history"] = data
    return log_data | tagged