from collections import defaultdict
from copy import copy

import numpy as np
import xarray as xr
import yaml
from toolz import merge


class SetupLoadError(RuntimeError):
    pass


class Setup:
    """

    Parameters
    ----------
    setup : dict, str or stream
        Parameters for initialization of setup.
    """

    def __init__(self, setup):
        if isinstance(setup, dict):
            setup_dict = setup
        else:
            setup_dict = yaml.safe_load(setup)

        self._qubits = {}
        self._gates = defaultdict(dict)

        self.name = setup_dict.get("name", "")
        self.dim = setup_dict.get('dim', 3)
        version = setup_dict.get("version", "1")
        if version != "1" and version != 1:
            raise SetupLoadError(
                "Unknown setup schema version: {}".format(version))
        self._load_setup_v1(setup_dict)

    def _load_setup_v1(self, setup_dict):
        params = copy(setup_dict.get("setup"))
        if params is None:
            raise SetupLoadError('Setup does not define "setup" section')

        for params_dict in params:
            qubits = tuple(params_dict.pop("qubits", tuple()))
            qubit = str(params_dict.pop("qubit", ""))
            if qubit:
                qubits = (qubit,)
            if qubits in self._qubits.keys():
                what = (
                    "Parameters for qubit(s) {}".format(", ".join(qubits))
                    if qubits
                    else "Default qubit parameters"
                )
                raise SetupLoadError(
                    what + " defined repeatedly in the setup.")
            self._qubits[qubits] = params_dict

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls(f)

    def to_dataset(self):
        qubit_set = set()
        pair_set = set()
        common_params = {}
        specific_params = defaultdict(set)
        shared_params = defaultdict(set)

        for qubits, params in self._qubits.items():
            if qubits == tuple():
                common_params.update(params)
            else:
                if len(qubits) == 1:
                    qubit_set.update(qubits)
                    for param in params.keys():
                        specific_params[param].update(qubits)
                else:
                    pair_set.add(qubits)
                    for param in params.keys():
                        shared_params[param].add(qubits)

        qubit_set = sorted(list(qubit_set))
        pair_set = sorted(list(pair_set))

        qubit_params = {}
        for param in specific_params.keys():
            _param_vals = []
            for qubit in qubit_set:
                if qubit in specific_params[param]:
                    _param_vals.append(self.param(param, qubit))
                elif param in common_params:
                    _param_vals.append(self.param(param, qubit))
                else:
                    _param_vals.append(np.nan)

            if param in common_params:
                del common_params[param]

            qubit_params[param] = (['qubit'], _param_vals)

        pair_params = {param: (
            'qubit_pair', [self.param(param, *qubits)
                           if qubits in shared_params[param] else np.nan
                           for qubits in pair_set])
                       for param in shared_params.keys()}

        dataset = xr.Dataset(
            data_vars=merge(
                qubit_params,
                pair_params
            ),
            coords=merge(
                {
                    'qubit': qubit_set,
                    'qubit_pair': ["{},{}".format(q1, q2) for q1, q2 in pair_set]
                },
                common_params)
        )

        return dataset

    def param(self, param, *qubits):
        try:
            return self._qubits[qubits][param]
        except KeyError:
            pass
        try:
            return self._qubits[tuple()][param]
        except KeyError:
            pass
        raise KeyError(
            'Parameter "{}" is not defined for qubit(s) {}'.format(
                param, ", ".join(qubits)
            )
        )

    def qubit_params(self, *qubits):
        return {**self._qubits[tuple()], **self._qubits[qubits]}
