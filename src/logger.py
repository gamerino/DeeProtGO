# ==============================================================================
# IBB-CONICET-UNER
# sinc(i)-CONICET-UNL. http://sinc.unl.edu.ar/
# G. Merino, et al.
# gmerino@ingenieria.uner.edu.ar
# gmerino@sinc.unl.edu.ar
# ==========================================================================

import os

class Logger:
    """Text logger"""
    def __init__(self, dir_name = "./"):
        """
        Args:
            dir_name (str): Path to the directory where log files are saved.
        """        
        self.out_dir = dir_name
        self.fout = dict()
        self.current_log = ""
        if not os.path.exists( dir_name ):
            os.mkdir( dir_name )
        global logger
        logger = self

    def start(self, file_name = "log"):
        self.fout[ file_name ] = open( self.out_dir + file_name + ".log", 'a')
        self.current_log = file_name

    def log(self, msg, log_name = None, verbose = True):
        if not log_name:
            log_name = self.current_log
        if log_name not in self.fout.keys():
            self.fout[ log_name ] = open( self.out_dir + log_name + ".log", 'a')
        self.fout[ log_name ].write( msg )
        self.fout[ log_name ].flush()
        if verbose:
            print(msg, end = '')

    def close(self, log_name = None):
        if log_name == None:
            for k in self.fout.keys():
                self.fout[ k ].close()
        else:
            self.fout[ log_name ].close()
