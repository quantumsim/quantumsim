import abc


class Backend(metaclass=abc.ABCMeta):
    """A metaclass, that defines standard interface for Quantumsim backend."""
    def __init__(self):
        raise NotImplementedError()
