from inspect import signature

import numpy as np
import xarray as xr

from ..operations import ParametrizedOperation
from ..circuits import deparametrize, FinalizedCircuit, _to_str
from ..states import State


class Controller:
    """
     The controller class handles the application of circuits to a single state. It automatically parses the circuit for free parameters set by the model and handles the data output of the circuit.

    Parameters
        ----------
        state : quantumsim.states.State
            The initial state of the system.
        circuits : dict
            A dictionary of circuit name and the corresponding quantumsim.circuits.FinalizedCircuit instances.
        rng : int or numpy.random.RandomState
            Either an integer number to seed a RandomState instance or an already initialized instance. The RandomState is used as the random generator for the functions within the experiment
    """

    def __init__(self, state, circuits, rng):
        if isinstance(rng, np.random.RandomState):
            self._rng = rng
        elif isinstance(rng, int):
            self._rng = np.random.RandomState(rng)
        else:
            raise ValueError(
                "Please provide a integer seed or an instance of a np.randomRandomState")

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

    def run(self, circuit_name, num_runs=1, **params):
        """
        run Applies the circuit corresponding to the provided name to the internal state stored by the controller.

        Parameters
        ----------
        circuit_name : str
            The name of the circuit stored by the controlled
        num_runs : int, optional
            The number of repeated applications of the given circuit, by default 1

        Returns
        -------
        xarray.DataArray or None
            If the circuit had any free parmeters, return the data array containing the realized parameter value of each parameter and over the repeated runs, else it returns None
        """
        try:
            circuit = self._circuits[circuit_name]
        except KeyError:
            raise KeyError("Circuit {} not found".format(circuit_name))

        # Extract all parameters, for which a callable expression was provided
        _cur_param_funcs = {par: params.pop(par) for par in list(
            params) if callable(params[par])}
        # Combine with the automatically generated ones,
        # overwriting any if the user has provided a different function
        param_funcs = {**circuit._param_funcs, **_cur_param_funcs}

        if params:
            # At this points params only contains the fixed parameters
            circuit = circuit(**params)

        unset_params = circuit.params - param_funcs.keys()
        if len(unset_params) != 0:
            raise KeyError(*unset_params)

        outcomes = []

        for _ in range(num_runs):
            outcome = self._run_circuit(circuit, param_funcs=param_funcs)
            if outcome is not None:
                outcomes.append(outcome)

        if outcomes:
            result = xr.concat(outcomes, dim='run')
            # Attach all parameters that had fixed values across the repeated applications
            for param, param_val in params.items():
                result[param] = param_val
            return result
        return None

    def _run_circuit(self, circuit, *, param_funcs=None):
        """
        _run_circuit Sequentally applies a finalized circuit to the internal state

        Parameters
        ----------
        circuit : quantumsim.circuits.FinalizedCircuit
            The finalized circuit to be applied
        param_funcs : dict, optional
            The dictionary of free parameter names and thier corresponding callable object that implement them, by default None

        Returns
        -------
        xarray.DataArray
            The data array containing the values of the realized free parameter values for this circuit
        """
        if len(circuit.params) != 0:
            outcome = xr.DataArray(
                dims=['param'],
                coords={'param': list(circuit.params)})
        else:
            outcome = None

        for operation, inds in circuit.operation.units():
            # Extract the indicies for this operation from the state
            op_qubits = [circuit.qubits[i] for i in inds]
            op_inds = [self._state.qubits.index(q) for q in op_qubits]

            if isinstance(operation, ParametrizedOperation):
                # Get the free parameters of the operation and evaluate them
                _op_params = _to_str(operation.params)
                _controller_params = {
                    'state': self._state.partial_trace(*op_qubits),
                    'rng': self._rng,
                    'outcome': outcome}

                _eval_params = {}
                for param in _op_params:
                    param_func = param_funcs[param]
                    sig = signature(param_func)
                    func_args = {par.name: _controller_params[par.name]
                                 for par in sig.parameters.values()
                                 if par.kind == par.POSITIONAL_OR_KEYWORD and
                                 par.name != 'self'}
                    _eval_params[param] = param_func(**func_args)

                # sub in the operation
                operation = deparametrize(operation, _eval_params)
                # append realized value to the pre-located DataArray
                outcome.loc[{'param': list(_op_params)}] = list(
                    _eval_params.values())

            # Apply each operation, which now should have all parameters fixed
            operation(self._state.pauli_vector, *op_inds)

            # If operation was not trace perserving, renormalize state.
            # Not sure if I should add this here?
            if not np.isclose(self._state.trace(), 1):
                self._state.renormalize()

        return outcome
