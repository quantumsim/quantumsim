# noinspection PyUnresolvedReferences,PyProtectedMember
from ._version import __version__
from .states import State

__all__ = []

for module in ['bases', 'operations', 'states', 'models']:
    exec('from . import {0}'.format(module))
    __all__.append(module)

del module


def test(verbose=True):
    from pytest import main
    from os.path import dirname, abspath, join
    return main([dirname(abspath(join(__file__, '..'))), "-s"] +
                (['-v'] if verbose else []))
