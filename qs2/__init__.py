# noinspection PyUnresolvedReferences,PyProtectedMember
from ._version import __version__
from .states import State

__all__ = [
    'bases',
    'operations',
    'State'
]


def test(verbose=True):
    from pytest import main
    from os.path import dirname, abspath, join
    return main([dirname(abspath(join(__file__, '..'))), "-s"] +
                (['-v'] if verbose else []))
