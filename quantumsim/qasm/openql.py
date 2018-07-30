import copy
import json
import numpy as np
import re
import warnings
from itertools import chain
from types import SimpleNamespace

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

    def parse(self, qasm, rng=None, time_start=0.):
        return list(self.gen_circuits(qasm, rng))

    def gen_circuits(self, qasm, rng=None, time_start=0.):
        rng = ct._ensure_rng(rng)
        if isinstance(qasm, str):
            return self._gen_circuits_fn(qasm, rng, time_start)
        else:
            return self._gen_circuits(qasm, rng, time_start)

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

    def _gen_circuits(self, fp, rng, time_start=0.):
        # Getting the initial statement with the number of qubits
        rng = ct._ensure_rng(rng)
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
            # yield self._parse_circuit(title, n_qubits, source, rng)
            # We pass the same rng to avoid correlations between measurements,
            # everything else must be re-initialized
            parse_state = self._init_parse_state(rng=rng, time=time_start)
            yield self._parse_circuit(parse_state, title, n_qubits, source)

    def _gen_circuits_fn(self, filename, rng, time_start=0):
        with open(filename, 'r') as fp:
            generator = self._gen_circuits(fp, rng, time_start)
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

    def _add_gate(self, parse_state, circuit, gate_spec, gate_label):
        try:
            gate, time_end = self._gate_spec_to_gate(
                gate_spec, gate_label, parse_state)
            circuit.add_gate(gate)
            parse_state.time_current = time_end
        except Exception as e:
            raise ConfigurationError(
                "Could not construct gate from gate_spec") from e

    @classmethod
    def _gate_spec_to_gate(cls, gate_spec, gate_label, parse_state):
        duration = gate_spec['duration']
        qubits = gate_spec['qubits']
        time_start = parse_state.time_current
        if cls._gate_is_ignored(gate_spec):
            return None, time_start
        elif cls._gate_is_measurement(gate_spec):
            seed = parse_state.rng.randint(1 << 32)
            gate = ct.Measurement(
                qubits[0],
                time_start+0.5*duration,
                ct.uniform_sampler(np.random.RandomState(seed=seed))
            )
            gate.label = r"$\circ\!\!\!\!\!\!\!\nearrow$"
            # return gate
        elif cls._gate_is_single_qubit(gate_spec):
            m = np.array(gate_spec['matrix'], dtype=float)
            # TODO: verify if it is not conjugate
            kraus = (m[:, 0] + m[:, 1]*1j).reshape((2, 2))
            gate = ct.SinglePTMGate(
                qubits[0],
                time_start+0.5*duration,
                ptm.single_kraus_to_ptm(kraus)
            )
            gate.label = gate_label
            # return gate
        elif cls._gate_is_two_qubit(gate_spec):
            m = np.array(gate_spec['matrix'], dtype=float)
            # TODO: verify if it is not conjugate
            kraus = (m[:, 0] + m[:, 1]*1j).reshape((4, 4))
            gate = ct.TwoPTMGate(
                qubits[0],
                qubits[1],
                ptm.double_kraus_to_ptm(kraus),
                time_start+0.5*duration,
            )
            gate.label = gate_label
            # return gate
        else:
            raise ConfigurationError(
                'Could not identify gate type from gate_spec')

        time_end = time_start + duration
        return gate, time_end

    def _parse_circuit(self, parse_state, title, n_qubits, source):
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
            self._add_gate(parse_state, circuit,
                           gate_spec=gate, gate_label=instr)
        circuit.add_waiting_gates(tmin=0, tmax=parse_state.time_current)
        circuit.order()
        return circuit

    @staticmethod
    def _gate_is_ignored(gate_spec):
        out = gate_spec['type'] == 'none'
        return out

    @staticmethod
    def _gate_is_measurement(gate_spec):
        out = gate_spec['type'] == 'readout'
        if out:
            if len(gate_spec['qubits']) != 1:
                raise NotSupportedError(
                    'Only single-qubit measurements are supported')
        return out

    @staticmethod
    def _gate_is_single_qubit(gate_spec):
        out = (gate_spec['type'] != 'readout') and \
              (len(gate_spec['qubits']) == 1)
        if out:
            if len(gate_spec['matrix']) != 4:
                raise ConfigurationError(
                    'Process matrix is incompatible with number of qubits')
        return out

    @staticmethod
    def _gate_is_two_qubit(gate_spec):
        out = (gate_spec['type'] != 'readout') and \
              (len(gate_spec['qubits']) == 2)
        if out:
            if len(gate_spec['matrix']) != 16:
                raise ConfigurationError(
                    'Process matrix is incompatible with number of qubits')
        return out

    @staticmethod
    def _init_parse_state(rng=None, time=0):
        return SimpleNamespace(rng=ct._ensure_rng(rng), time_current=time)
