# noinspection PyUnresolvedReferences,PyProtectedMember
from ._version import __version__
from . import pauli_vectors, bases
from .operations import Operation
from .setups import Setup
from .states import State
from .models import Model
from .controllers import Controller

PauliVector = pauli_vectors.Default

__all__ = [
    'bases',
    'pauli_vectors',
    'Model',
    'Operation',
    'PauliVector',
    'Setup',
    'State',
    'Controller',
]


def test(verbose=True):
    from pytest import main
    from os.path import dirname, abspath, join
    return main([dirname(abspath(join(__file__, '..'))), "-s"] +
                (['-v'] if verbose else []))
