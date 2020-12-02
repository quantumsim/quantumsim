# noinspection PyProtectedMember
from ._version import __version__
from .bases import PauliBasis
from .circuits import Gate, Box, Circuit, FinalizedCircuit
from .setups import Setup
from .states import State, StateNumpy
from .models import Model, perfect_qubits as gates
from .controllers import Controller

__all__ = [
    'PauliBasis',
    'Model',
    'Setup',
    'State',
    'StateNumpy',
    'Controller',
    'gates',
]

try:
    # noinspection PyUnresolvedReferences
    from .states import StateCuda
    __all__ += 'StateCuda'
    State = StateCuda
except ImportError:
    State = StateNumpy


def test(verbose=True):
    from pytest import main
    from os.path import dirname, abspath, join
    return main([dirname(abspath(join(__file__, '..'))), "-s"] +
                (['-v'] if verbose else []))
