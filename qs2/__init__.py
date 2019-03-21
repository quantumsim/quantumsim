# noinspection PyUnresolvedReferences,PyProtectedMember
from ._version import __version__
from . import states
from .operations.operation import Operation

State = states.Default

__all__ = [
    'bases',
    'states',
    'State',
    'Operation',
]


def test(verbose=True):
    from pytest import main
    from os.path import dirname, abspath, join
    return main([dirname(abspath(join(__file__, '..'))), "-s"] +
                (['-v'] if verbose else []))
