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

        version = setup_dict.get('version', '1')
        if version != '1' and version != 1:
            raise SetupLoadError('Unknown setup schema version: {}'
                                 .format(version))
        self._load_setup_v1(setup_dict)

    def _load_setup_v1(self, setup_dict):
        qubits = copy(setup_dict.get('qubits'))
        if qubits is None:
            raise SetupLoadError('Setup does not define "qubits" section')

        for qubit_dict in qubits:
            name = str(qubit_dict.pop('name', ''))
            if name in self._qubits.keys():
                what = 'Parameters for qubit "{}"'.format(name) if name else \
                    'Default qubit parameters'
                raise SetupLoadError(what + ' defined repeatedly in the setup.')
            self._qubits[name] = qubit_dict

        gates = copy(setup_dict.get('gates'))
        if gates is None:
            raise SetupLoadError('Setup does not define "gates" section')

        for gate_dict in gates:
            name = gate_dict.pop('name', None)
            if not name:
                raise SetupLoadError('All gate declarations in the setup must '
                                     'include a field "name".')
            qubits = gate_dict.pop('qubits', [])
            if isinstance(qubits, list):
                qubits = tuple(qubits)
            elif isinstance(qubits, str):
                qubits = (qubits,)
            else:
                raise SetupLoadError('"qubits" keyword in a gate definition '
                                     'must be either a string, or a list.')
            if qubits in self._gates[name].keys():
                what = ('Default parameters for gate "{}"'.format(name)
                        if len(qubits) == 0 else
                        'Parameters for gate "{}" on qubits {}'
                        .format(name, ", ".join(qubits)))
                raise SetupLoadError(what +
                                     ' are defined repeatedly in the setup.')
            self._gates[name][qubits] = gate_dict

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            return cls(f)

