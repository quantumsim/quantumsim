import numpy as np
import xarray as xr

from ..operations import ParametrizedOperation
from ..circuits import deparametrize, FinalizedCircuit, _to_str
from ..states import State


class Controller:
    def __init__(self, state, circuits, rng=None, circuit_params=None):
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

    @property
    def state(self):
        return self._state

    @property
    def circuits(self):
        return self._circuits

    def apply(self, circuit_name, num_runs=1, **params):
        try:
            circuit = self._circuits[circuit_name](**params)
        except KeyError:
            raise KeyError("Circuit {} not found".format(circuit_name))

        unset_params = circuit.params - circuit._param_funcs.keys()
        if len(unset_params) != 0:
            raise KeyError(*unset_params)

        outcomes = []

        for _ in range(num_runs):
            outcome = self._apply_circuit(circuit)
            if outcome is not None:
                outcomes.append(outcome)

        if outcomes:
            result = xr.concat(outcomes, dim='run')
            for param, param_val in params.items():
                result[param] = param_val
            return result
        return None

    def _apply_circuit(self, circuit):
        if len(circuit.params) != 0:
            outcome = xr.DataArray(
                dims=['param'],
                coords={'param': list(circuit.params)})
        else:
            outcome = None

        for operation, inds in circuit.operation.units():
            op_qubits = [circuit.qubits[i] for i in inds]
            op_inds = [self._state.qubits.index(q) for q in op_qubits]

            if isinstance(operation, ParametrizedOperation):
                _op_params = _to_str(operation.params)
                _eval_params = {
                    param: circuit._param_funcs[param](
                        state=self._state.partial_trace(*op_qubits),
                        rng=self._rng,
                        outcome=outcome)
                    for param in _op_params}

                operation = deparametrize(operation, _eval_params)

                outcome.loc[{'param': list(_op_params)}] = list(
                    _eval_params.values())

            operation(self._state.pauli_vector, *op_inds)

            if not np.isclose(self._state.trace(), 1):
                self._state.renormalize()

        return outcome
