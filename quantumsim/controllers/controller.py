import xarray as xr

from ..circuits import FinalizedCircuit
from ..states import State


class Controller:
    def __init__(self, state):
        self._state = state

    @property
    def state(self):
        return self._state

    def run_round(self, num_rounds=1):
        raise NotImplementedError

    def _apply_circuit(self, circuit_name, **kwargs):
        outcome = xr.DataArray()
        return outcome
