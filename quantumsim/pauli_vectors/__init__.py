from .numpy import PauliVectorNumpy

__all__ = ['Default', 'PauliVectorNumpy']

try:
    from .cuda import PauliVectorCuda
    __all__.append('PauliVectorCuda')
    Default = PauliVectorCuda
except ImportError:
    import warnings
    warnings.warn('Could not import CUDA backend. Either PyCUDA is not '
                  'installed, or your PC has no NVidia GPU at all. Be wise '
                  'with a difficulty of the problem you state to Quantumsim.')
    Default = PauliVectorNumpy
