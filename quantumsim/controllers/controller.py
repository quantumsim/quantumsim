from itertools import chain

import numpy as np
import xarray as xr

from ..operations import ParametrizedOperation, Operation
from ..circuits import TimeAwareCircuit, TimeAgnosticCircuit, FinalizedCircuit
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

        self._free_params = dict.fromkeys(chain(*_free_circ_params), None)

        self._circuits = {circ_name: circ.finalize()
                          for circ_name, circ in circuits.items()}

        self._params = circuit_params

    @property
    def state(self):
        return self._state

    @property
    def circuits(self):
        return self._circuits

    def eval_params(self, params):
        raise NotImplementedError

    def run_round(self):
        raise NotImplementedError

    def _apply_circuit(self, circuit_name, **kwargs):
        try:
            circuit = self._circuits[circuit_name]
        except KeyError:
            raise KeyError("Circuit {} not found".format(circuit_name))

        if len(circuit.params) > 0:
            missing_params = circuit.params - kwargs.keys()
            if len(missing_params) != 0:
                unset_params = missing_params - self._params.keys()
                if len(unset_params) != 0:
                    raise KeyError(**unset_params)
                units, qubits = [], []
                for operation, inds in circuit.operation.units():
                    if isinstance(operation, ParametrizedOperation):
                        if units:
                            sub_circ = FinalizedCircuit(
                                Operation.from_sequence(units).compile(),
                                qubits)
                            sub_circ.apply_to(self._state)
                            units, qubits = [], []
                        _params = self.eval_params(missing_params)
                        # TODO: Once previous operation are compiles, the paramterizerd op can be evaluated
                        # FIXME: One doesn't need to eval all missing parm
                        # FIXME: deparameterize doesn't need to be calss method - take it outside and import here. Alternative solution - make circuitm and call, but useless bloat.
                        circuit._deparametrize(operation, _params).at(*inds)
                    else:
                        units.append(operation)
                        qubits += [circuit.qubits[i]
                                   for i in inds if circuit.qubits[i] not in qubits]
            else:
                circuit(**kwargs).apply_to(self._state)
        else:
            circuit.apply_to(self._state)
