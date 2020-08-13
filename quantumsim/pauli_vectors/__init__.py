from .numpy import PauliVectorNumpy, PauliVectorBase

__all__ = ['Default', 'PauliVectorNumpy', 'PauliVectorBase', ]

try:
    from .opencl import PauliVectorOpenCL
    __all__.append('PauliVectorOpenCL')
    Default = PauliVectorOpenCL
except ImportError:
    Default = PauliVectorNumpy

try:
    from .cuda import PauliVectorCuda
    __all__.append('PauliVectorCuda')
    Default = PauliVectorCuda
except ImportError:
    pass

if Default == PauliVectorNumpy:
    import warnings
    warnings.warn('Could not initialize any of GPU backends. If you need '
                  'high-performance computations, check that either PyOpenCL or PyCUDA '
                  'are accessible and operational.')