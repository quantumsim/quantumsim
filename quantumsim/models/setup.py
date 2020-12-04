import yaml
from collections import defaultdict

from copy import copy


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
        version = setup_dict.get("version", "1")
        if version != "1" and version != 1:
            raise SetupLoadError("Unknown setup schema version: {}".format(version))
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
                raise SetupLoadError(what + " defined repeatedly in the setup.")
            self._qubits[qubits] = params_dict

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls(f)

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
                param, ", ".join((str(q) for q in qubits))
            )
        )

    def qubit_params(self, *qubits):
        return {**self._qubits[tuple()], **self._qubits[qubits]}
