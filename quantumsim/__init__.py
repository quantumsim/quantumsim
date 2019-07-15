# noinspection PyUnresolvedReferences,PyProtectedMember
from ._version import __version__
from . import pauli_vectors, bases
from .operations.operation import Operation

PauliVector = pauli_vectors.Default

__all__ = [
    'bases',
    'pauli_vectors',
    'PauliVector',
    'Operation',
]


def test(verbose=True):
    from pytest import main
    from os.path import dirname, abspath, join
    return main([dirname(abspath(join(__file__, '..'))), "-s"] +
                (['-v'] if verbose else []))
