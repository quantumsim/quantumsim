from .numpy import State, StateNumpy

__all__ = ['State', 'StateNumpy']

try:
    from .cuda import StateCuda
    __all__.append('StateCuda')
except ImportError:
    import warnings
    warnings.warn('Could not import CUDA backend. Either PyCUDA is not '
                  'installed, or your PC has no NVidia GPU at all. Be wise '
                  'with a difficulty of the problem you state to Quantumsim.')
