from collections.abc import Iterable
from inspect import signature

import numpy as np
import xarray as xr

from .. import State
from ..circuits import FinalizedCircuit, _to_str, deparametrize
from ..operations import ParametrizedOperation


class Controller:
    """
     Experiment controller

     The controller handles the application of circuits to a single state.
     It automatically parses the circuit for free parameters set by the model and handles the data output of the circuit.

    Parameters
        ----------
        state : quantumsim.states.State
            The initial state of the system.
        circuits : dict
            A dictionary of circuit name and the corresponding quantumsim.circuits.FinalizedCircuit instances.
        rng : int or numpy.random.RandomState
            Either an integer number to seed a RandomState instance or an already initialized instance. The RandomState is used as the random generator for the functions within the experiment
    """

    def __init__(self, circuits, parameters=None):
        if not isinstance(circuits, dict):
            raise ValueError(
                "Circuits expected to be dict instance, instead provided as {}".format(
                    type(circuits)
                )
            )
        if not all(
            isinstance(circ_name, str) and isinstance(circ, FinalizedCircuit)
            for circ_name, circ in circuits.items()
        ):
            raise ValueError(
                "The circuit dictionary should contain the names of circuits as the keys and the finalized circuits as the values."
            )

        self._circuits = circuits

        qubits = set()
        for circ in self.circuits.values():
            qubits.update(circ.qubits)
        self._qubits = qubits

        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError(
                    "Parameters expected to be dict instance, instead provided as {}".format(
                        type(parameters)
                    )
                )

        self._parameters = parameters or {}

        self._rng = None
        self._state = None

    @property
    def state(self):
        return self._state

    @property
    def circuits(self):
        return self._circuits

    def seed(self, seed):
        if not isinstance(seed, int):
            raise ValueError("The provided seed must be an intereger num")
        self._rng = np.random.RandomState(seed)

    def prepare_state(self, method=None):
        if method is not None:
            self._state = method(self._qubits)
        else:
            self._state = State(self._qubits)

    def run(self, experiment, seed):
        if not callable(experiment):
            raise ValueError("The experiment must be a defined function")

        outcomes = []

        if isinstance(seed, int):
            seed_sequence = [seed]
        elif isinstance(seed, Iterable):
            seed_sequence = seed
        else:
            raise ValueError(
                "Seed expected to be integer or iterable sequence of integers,instead provided {}".format(
                    type(seed)
                )
            )

        for seed_val in seed_sequence:
            self.seed(seed_val)

            exp_outcomes = experiment()

            if exp_outcomes is not None:
                if not isinstance(exp_outcomes, list):
                    raise ValueError(
                        "The outcome of the experiment should be a list of the circuit outputs"
                    )
                if any(outcome is None for outcome in exp_outcomes):
                    raise ValueError("The outcome list should not contain any None")

                exp_outcome_dataset = xr.merge(exp_outcomes)
                exp_outcome_dataset["seed"] = seed_val
                outcomes.append(exp_outcome_dataset)

        if len(outcomes) > 0:
            return xr.concat(outcomes, dim="seed")
        return None

    def apply(self, circuit_name, num_runs=1, **params):
        """
        Apply the circuit corresponding to the provided name to the internal state stored by the controller.

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

        if self._state is None:
            raise ValueError(
                "A state must be initialized before circuit application is possible"
            )

        # Extract all parameters, for which a callable expression was provided

        given_params = {**self._parameters, **params}

        _cur_param_funcs = {
            par: given_params.pop(par)
            for par in list(given_params)
            if callable(given_params[par])
        }

        # Combine with the automatically generated ones,
        # overwriting any if the user has provided a different function
        param_funcs = {**circuit._param_funcs, **_cur_param_funcs}

        if self._rng_required(param_funcs) and self._rng is None:
            raise ValueError(
                "A random number generator must be initialized, please seed the controller"
            )

        if given_params:
            # At this points params only contains the fixed parameters
            circuit = circuit(**given_params)

        unset_params = circuit.params - param_funcs.keys()
        if len(unset_params) != 0:
            raise KeyError(*unset_params)

        outcomes = []

        for _ in range(num_runs):
            outcome = self._apply_circuit(circuit, param_funcs=param_funcs)
            if outcome is not None:
                outcomes.append(outcome.rename({"param": circuit_name + "_param"}))

        if outcomes:
            result = xr.concat(outcomes, dim=circuit_name + "_run")
            # Add fixed parameters
            for param, param_val in params.items():
                result[param] = param_val
            result.name = circuit_name
            return result
        return None

    def _apply_circuit(self, circuit, *, param_funcs=None):
        """
        _run_circuit Sequentally applies a finalized circuit to the internal state

        Parameters
        ----------
        circuit : quantumsim.circuits.FinalizedCircuit
            The finalized circuit to be applied
        param_funcs : dict, optional
            The dictionary of free parameter names and their corresponding callable object that implement them, by default None

        Returns
        -------
        xarray.DataArray
            The data array containing the values of the realized free parameter values for this circuit
        """
        if len(circuit.params) != 0:
            outcome = xr.DataArray(
                dims=["param"], coords={"param": list(circuit.params)}
            )
        else:
            outcome = None

        for operation, inds in circuit.operation.units():
            # Extract the indices for this operation from the state
            op_qubits = [circuit.qubits[i] for i in inds]
            op_inds = [self._state.qubits.index(q) for q in op_qubits]

            if isinstance(operation, ParametrizedOperation):
                # Get the free parameters of the operation and evaluate them
                _op_params = _to_str(operation.params)
                _controller_params = {
                    "state": self._state.partial_trace(*op_qubits),
                    "rng": self._rng,
                    "outcome": outcome,
                }

                _eval_params = {}
                for param in _op_params:
                    param_func = param_funcs[param]
                    sig = signature(param_func)
                    func_args = {
                        par.name: _controller_params[par.name]
                        for par in sig.parameters.values()
                        if par.kind == par.POSITIONAL_OR_KEYWORD and par.name != "self"
                    }
                    _eval_params[param] = param_func(**func_args)

                # sub in the operation
                operation = deparametrize(operation, _eval_params)
                # append realized value to the pre-located DataArray
                outcome.loc[{"param": list(_op_params)}] = list(_eval_params.values())

            # Apply each operation, which now should have all parameters fixed
            operation(self._state.pauli_vector, *op_inds)

            # If operation was not trace perserving, renormalize state.
            # Not sure if I should add this here?
            if not np.isclose(self._state.trace(), 1):
                self._state.renormalize()

        return outcome

    def _rng_required(self, param_funcs):
        for func in param_funcs.values():
            sig = signature(func)
            if "rng" in list(sig.parameters):
                return True
        return False
