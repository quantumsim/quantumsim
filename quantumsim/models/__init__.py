from . import qubits
from . import transmons
from .model import Model

__all__ = ['Model', 'qubits', 'transmons']

import warnings
warnings.warn(
    "quantumsim.models module in current state is a temporary stop-gap\n"
    "solution and will be redesigned (under current or another name).\n"
    "We don't promise the future compatibility."
)
