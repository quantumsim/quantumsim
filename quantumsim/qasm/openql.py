import copy
import json
import numpy as np
import re
import warnings
from itertools import chain

import quantumsim.circuit as ct
import quantumsim.ptm as ptm


class ConfigurationError(RuntimeError):
    pass


class QasmError(RuntimeError):
    pass


class NotSupportedError(RuntimeError):
    pass


class OpenqlParser:

    def __init__(self, config):
        if isinstance(config, str):
            with open(config, 'r') as c:
                self._cfg = json.load(c)
        else:
            self._cfg = config

        try:
            self._instr = self._cfg['instructions']
        except KeyError:
            raise ConfigurationError(
                'Could not find "instructions" block in config')

        self._simulation_settings = self._cfg.get('simulation_settings', None)

    def parse(self, qasm, rng=None):
        return list(self.gen_circuits(qasm, rng))

    def gen_circuits(self, qasm, rng=None):
        rng = ct._ensure_rng(rng)
        if isinstance(qasm, str):
            return self._gen_circuits_fn(qasm, rng)
        else:
            return self._gen_circuits(qasm, rng)

    @staticmethod
    def _gen_circuits_src(fp):
        circuit_title = None
        circuit_src = []
        for line_full in fp:
            line = line_full.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('.'):
                if circuit_title:
                    yield circuit_title, circuit_src
                circuit_title = line[1:]
                circuit_src = []
            else:
                circuit_src.append(line)
        if circuit_title:
            yield circuit_title, circuit_src
        else:
            warnings.warn("Could not find any circuits in the QASM file.")

    def _gen_circuits(self, fp, rng):
        # Getting the initial statement with the number of qubits
        qubits_re = re.compile(r'^\s*qubits\s+(\d+)')
        n_qubits = None

        for line in fp:
            m = qubits_re.match(line)
            if m:
                n_qubits = int(m.groups()[0])
                break

        if not n_qubits:
            raise QasmError('Number of qubits is not specified')

        for title, source in self._gen_circuits_src(fp):
            yield self._parse_circuit(title, n_qubits, source, rng)

    def _gen_circuits_fn(self, filename, rng):
        with open(filename, 'r') as fp:
            generator = self._gen_circuits(fp, rng)
            for circuit in generator:
                yield circuit

    def _add_qubit(self, circuit, qubit_name):
        if self._simulation_settings:
            params = copy.deepcopy(self._simulation_settings.get(qubit_name))
            if not params:
                raise ConfigurationError(
                    'Could not find simulation settings for qubit {}'
                    .format(qubit_name))
            em = params.pop('error_model', None)
            if em == 't1t2':
                circuit.add_qubit(qubit_name, **params)
            else:
                raise ConfigurationError(
                    'Unknown error model for qubit "{}": "{}"'
                    .format(qubit_name, em))
        else:
            # No simulation settings provided -- assuming ideal qubits
            circuit.add_qubit(qubit_name)

    def _add_gate(self, circuit, label, gate_spec, rng):
        time = 0   # TODO !!!
        try:
            gate = self._gate_spec_to_gate(gate_spec, time, label, rng)
            circuit.add_gate(gate)
        except Exception as e:
            raise ConfigurationError(
                "Could not construct gate from gate_spec") from e

    def _gate_spec_to_gate(self, gate_spec, time, label, rng):
        duration = gate_spec['duration']
        qubits = gate_spec['qubits']
        if self._gate_is_measurement(gate_spec):
            seed = rng.randint(1<<32)
            gate = ct.Measurement(
                qubits[0],
                time+0.5*duration,
                ct.uniform_sampler(np.random.RandomState(seed=seed))
            )
            gate.label = r"$\circ\!\!\!\!\!\!\!\nearrow$"
            return gate
        elif self._gate_is_single_qubit(gate_spec):
            m = np.array(gate_spec['matrix'], dtype=float)
            kraus = (m[:, 0] + m[:, 1]*1j).reshape((2, 2)) # TODO: verify if it is not conjugate
            gate = ct.SinglePTMGate(
                qubits[0],
                time+0.5*duration,
                ptm.single_kraus_to_ptm(kraus)
            )
            gate.label = label
            return gate
        elif self._gate_is_two_qubit(gate_spec):
            m = np.array(gate_spec['matrix'], dtype=float)
            kraus = (m[:, 0] + m[:, 1]*1j).reshape((4, 4)) # TODO: verify if it is not conjugate
            return ct.TwoPTMGate(
                qubits[0],
                qubits[1],
                ptm.double_kraus_to_ptm(kraus),
                time+0.5*duration,
            )
            gate.label = label
            return gate
        else:
            raise ConfigurationError('Could not identify gate type from gate_spec')

    def _parse_circuit(self, title, n_qubits, source, rng):
        gates = [self._instr[line] for line in source]
        qubits = set(chain(*(gate['qubits'] for gate in gates)))
        if len(qubits) > n_qubits:
            raise QasmError('Too many qubits in circuit .{}: '
                            '{} declared in the file header, '
                            '{} actually present.'
                            .format('a', 2, 3))

        circuit = ct.Circuit(title)
        for qubit in qubits:
            self._add_qubit(circuit, qubit)

        for instr, gate in zip(source, gates):
            self._add_gate(circuit, instr, gate, rng)

        return circuit

    @staticmethod
    def _gate_is_measurement(gate_spec):
        out = gate_spec['type'] == 'readout'
        if out:
            if len(gate_spec['qubits']) != 1:
                raise NotSupportedError('Only single-qubit measurements are supported')
        return out

    @staticmethod
    def _gate_is_single_qubit(gate_spec):
        out = (gate_spec['type'] != 'readout') and (len(gate_spec['qubits']) == 1)
        if out:
            if len(gate_spec['matrix']) != 4:
                raise ConfigurationError('Process matrix is incompatible with number of qubits')
        return out

    @staticmethod
    def _gate_is_two_qubit(gate_spec):
        out = (gate_spec['type'] != 'readout') and (len(gate_spec['qubits']) == 2)
        if out:
            if len(gate_spec['matrix']) != 16:
                raise ConfigurationError('Process matrix is incompatible with number of qubits')
        return out
