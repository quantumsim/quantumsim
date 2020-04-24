from collections import defaultdict
from collections.abc import Iterable
from inspect import signature
from itertools import count

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
            raise TypeError(
                "Circuits expected to be dict instance, instead provided as {}".format(type(circuits)))
        if not circuits:
            raise ValueError(
                "At least one circuit is expected, received empty dictionary instead")
        if not all(isinstance(circ_name, str) and isinstance(circ, FinalizedCircuit)
                   for circ_name, circ in circuits.items()):
            raise TypeError(
                "The circuits dictionary should be made up of quantumsim.FinalizedCircuit instance and string as their keys.")

        self._circuits = circuits

        self._qubits = set([qubit for circ in circuits.values()
                            for qubit in circ.qubits])

        if parameters is not None:
            if not isinstance(parameters, dict):
                raise TypeError(
                    "Parameters expected to be dict instance, instead provided as {}".format(type(parameters)))

        self._parameters = parameters or {}

        self._rng = None
        self._state = None
        self._outcomes = defaultdict(list)

    @property
    def state(self):
        return self._state

    @property
    def circuits(self):
        return self._circuits

    def set_rng(self, seed):
        if not isinstance(seed, int):
            raise TypeError(
                "Seed expected as int, provided as {} instead".format(type(seed)))
        self._rng = np.random.RandomState(seed)

    def prepare_state(self, dim=2):
        self._state = State(self._qubits, dim=dim)

    def to_dataset(self, array, concat_dim=None):
        if array is not None:
            if not isinstance(array, (xr.DataArray, xr.Dataset)):
                raise TypeError(
                    "array expected as xarray.DataArray or xarray.Dataset instance, instead given {}".format(type(array)))
            if array.name is None:
                raise ValueError("array must be named")
            self._outcomes[array.name].append(
                array.assign_attrs(concat_dim=concat_dim))

    def get_dataset(self, clear_cache=False):
        if len(self._outcomes) == 0:
            return None

        dataset = xr.Dataset()

        for circ_name, circ_outcomes in self._outcomes.items():
            grouped_outcomes = defaultdict(list)
            _placeholder_key = count()
            for out in circ_outcomes:
                concat_dim = out.concat_dim or next(_placeholder_key)
                grouped_outcomes[concat_dim].append(out)

            _suffix = count(1)
            for dim, grouped_arrays in grouped_outcomes.items():
                if isinstance(dim, str):
                    data_array = xr.concat(grouped_arrays, dim=dim)
                else:
                    # This is the case of a placaehold outcome inserted by us
                    data_array = grouped_arrays[0]

                if circ_name not in dataset:
                    dataset[circ_name] = data_array
                else:
                    dataset[circ_name + "_" + str(next(_suffix))] = data_array

        if clear_cache:
            self._outcomes = defaultdict(list)
        return dataset

    def run(self, run_experiment, seed, **parameters):
        if not callable(run_experiment):
            raise ValueError("The experiment must be a defined function")

        if isinstance(seed, int):
            seed_sequence = [seed]
        elif isinstance(seed, Iterable):
            seed_sequence = seed
        else:
            raise ValueError(
                "Seed expected to be integer or iterable sequence of integers,instead provided {}".format(type(seed)))

        datasets = []
        for seed_val in seed_sequence:
            self.set_rng(seed_val)
            run_experiment(**parameters)
            dataset = self.get_dataset(clear_cache=True)
            if dataset:
                dataset["seed"] = seed_val
                datasets.append(dataset)

        if len(seed_sequence) > 1:
            return xr.concat(datasets, dim="seed")
        elif len(seed_sequence) == 1:
            return datasets.pop(0)
        return None

    def apply(self, circuit_name, **parameters):
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

        array_coords = parameters.keys() - circuit.params

        # Extract all parameters, for which a callable expression was provided
        set_params = {**self._parameters, **parameters}

        set_param_funcs = {par: set_params.pop(par)
                           for par in list(set_params) if callable(set_params[par])}

        # Combine with the automatically generated ones,
        # overwriting any if the user has provided a different function
        param_funcs = {**circuit._param_funcs, **set_param_funcs}

        if self._rng_required(param_funcs) and self._rng is None:
            raise AttributeError(
                "Circuit requires a random number generator, but one was not initialized")

        if set_params:
            # At this points params only contains the fixed parameters
            circuit = circuit(**set_params)

        unset_params = circuit.params - param_funcs.keys()
        if len(unset_params) != 0:
            raise KeyError(*unset_params)

        outcome = self._apply_circuit(circuit, param_funcs=param_funcs)

        if outcome is not None:
            outcome.name = circuit_name
            for param in array_coords:
                outcome[param] = parameters[param]

        return outcome

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
                    "inds": op_inds,
                    "state": self._state,
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
                outcome.loc[{"param": list(_op_params)}] = list(
                    _eval_params.values())

            # Apply each operation, which now should have all parameters fixed
            operation(self._state.pauli_vector, *op_inds)

            # If operation was not trace perserving, renormalize state.
            # Not sure if I should add this here?
            # if not np.isclose(self._state.trace(), 1):
            #    self._state.renormalize()

        return outcome

    def _rng_required(self, param_funcs):
        for func in param_funcs.values():
            sig = signature(func)
            if "rng" in list(sig.parameters):
                return True
        return False
