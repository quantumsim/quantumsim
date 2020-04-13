import pytest

from numpy import pi
import xarray as xr
from pytest import approx
from quantumsim import Controller
from quantumsim import bases
from quantumsim.circuits import FinalizedCircuit, Circuit

single_qubit_bases = (bases.general(2),)
two_qubit_bases = single_qubit_bases * 2


# noinspection PyTypeChecker
class TestController:
    def test_input(self):
        pass
