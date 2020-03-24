from itertools import chain

import xarray as xr

from ..circuits import TimeAwareCircuit, TimeAgnosticCircuit
from ..states import State


class Controller:
    # TODO: Add state initialization methods

    def __init__(self, state, circuits, circuit_params):
        if not isinstance(state, State):
            raise ValueError("Please provide an initial state")
        self._state = state

        if not isinstance(circuits, dict):
            raise ValueError("circuits should be a dictionary")
        if not all(isinstance(circ_name, str) for circ_name in circuits.keys()):
            raise ValueError("Only names")
        if not all(isinstance(circ, (TimeAwareCircuit, TimeAgnosticCircuit))
                   for circ in circuits.values()):
            raise ValueError("Only circuits")

        _free_circ_params = list(
            circuit.free_parameters for circuit in circuits.values())

        # Common pareters in different circuits currently not allowed
        common_params = set.intersection(*_free_circ_params)
        if len(common_params) > 0:
            # TODO: add a better message ;D
            raise RuntimeError("ERROR")

        self._params = dict.fromkeys(chain(*_free_circ_params), None)

        self._circuits = {circ_name: circ.finalize()
                          for circ_name, circ in circuits.items()}

    @property
    def state(self):
        return self._state

    def run_round(self):
        raise NotImplementedError

    def _apply_circuit(self, circuit_name, **kwargs):
        try:
            self._circuits[circuit_name] @ self._state
        except KeyError:
            raise KeyError("Message")
        outcome = xr.DataArray()
        return outcome
