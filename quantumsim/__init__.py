# noinspection PyUnresolvedReferences,PyProtectedMember
from ._version import __version__
from . import states, bases
from .operations.operation import Operation
from .abstracts import Circuit, SimpleCircuit, QubitRegister

State = states.Default

__all__ = [
    'bases',
    'states',
    'State',
    'Operation',
    'Circuit',
    'SimpleCircuit',
    'QubitRegister'
]


def test(verbose=True):
    from pytest import main
    from os.path import dirname, abspath, join
    return main([dirname(abspath(join(__file__, '..'))), "-s"] +
                (['-v'] if verbose else []))
