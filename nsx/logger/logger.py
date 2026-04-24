''' Logging module '''
import logging
from mpi4py import MPI


class ParallelFilter(logging.Filter):
    ''' Suppress output on all procs but rank 0 '''

    def filter(self, record):
        ''' Filter out if not on rank 0 '''
        if MPI.COMM_WORLD.Get_rank() == 0:
            return True
        else:
            return False


ch = logging.StreamHandler()
ch.addFilter(ParallelFilter())
formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
ch.setFormatter(formatter)


class LoggerBase:
    ''' Logging base class, inherited by all classes the output of whose
    methods should be logged.
    The child class should call :py:meth:`~.set_log_filehandler` to specify the
    log file path.
    The :py:meth:`~.close_logs` method can be explicitly called to close the
    logs. However, it is preferable to use context managers for proper
    handling if a sequence of logged objects is created.

    Example:

    .. code:: python

        class solver(LoggerBase):
            def __init__(self, input):
                ...

        with solver(input) as sol:
            sol.compute_stuff()
    '''

    def __init__(self, **kwargs):
        ''' Initialize '''
        self.init_logging()

    # we cannot know when __del__ is actually called. use close_logs explicitly
    # or use context managers. Context managers clash with __del__, logs
    # possibly of subsequent logger objects will be deleted too
    # def __del__(self):
    #     self.close_logs()

    def __enter__(self):
        ''' Enter context manager '''
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ''' Close logs upon exiting a context manager. '''
        self.close_logs()

    def init_logging(self):
        ''' Create logger and handler '''
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(ch)
        self._logging_filehandler = None

    def set_log_filehandler(self, filepath):
        ''' Create logging File Handler '''
        fh = logging.FileHandler(filepath, 'a')
        fh.setFormatter(formatter)
        fh.addFilter(ParallelFilter())
        self._logging_filehandler = fh
        self.logger.addHandler(fh)

    def close_logs(self):
        ''' Close logs by removing all handlers '''
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
