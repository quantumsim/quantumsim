from math import isclose

import numpy as np
import xarray as xr

from ..operations import ParametrizedOperation
from ..circuits import deparametrize, FinalizedCircuit
from ..states import State


class Controller:
    # TODO: Add state initialization methods

    def __init__(self, state, circuits, circuit_params, rng=None):

        if isinstance(rng, np.random.RandomState):
            self._rng = rng
        elif isinstance(rng, int):
            self._rng = np.random.RandomState(rng)
        else:
            raise ValueError(
                "Please provide a seed or an instance of a np.randomRandomState")

        if not isinstance(state, State):
            raise ValueError("Please provide an initial state")
        self._state = state

        if not isinstance(circuits, dict):
            raise ValueError("circuits should be a dictionary")
        if not all(isinstance(circ_name, str) and isinstance(circ, FinalizedCircuit)
                   for circ_name, circ in circuits.items()):
            raise ValueError("Only names")

        self._circuits = circuits

        self._params = circuit_params
        self._outcome = xr.Dataset()

    @property
    def state(self):
        return self._state

    @property
    def circuits(self):
        return self._circuits

    @property
    def outcome(self):
        return self._outcome

    def apply(self, circuit_name, num_rounds=1, **params):
        try:
            circuit = self._circuits[circuit_name](**params)
        except KeyError:
            raise KeyError("Circuit {} not found".format(circuit_name))

        unset_params = circuit.params - self._params.keys()
        if len(unset_params) != 0:
            raise KeyError(*unset_params)
        for i in range(num_rounds):
            self._apply_circuit(circuit)

    def _apply_circuit(self, circuit):
        for operation, inds in circuit.operation.units():
            op_qubits = [circuit.qubits[i] for i in inds]
            op_inds = [self._state.qubits.index(q) for q in op_qubits]

            if isinstance(operation, ParametrizedOperation):
                _params = {
                    param: self._params[param](
                        state=self._state.partial_trace(*op_qubits),
                        rng=self._rng)
                    for param in operation.params}
                operation = deparametrize(operation, _params)

            operation(self._state.pauli_vector, *op_inds)

            if not isclose(self._state.trace(), 1):
                self._state.renormalize()
