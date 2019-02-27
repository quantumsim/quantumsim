from .backend import State

try:
    from .cuda import DensityMatrix as State
except ImportError:
    import warnings
    warnings.warn('Could not import CUDA backend. Either PyCUDA is not '
                  'installed, or your PC has no NVidia GPU. Be wise with a '
                  'difficulty of the problem you state to Quantumsim.')
    from .numpy import DensityMatrix as State
