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
        self._gates_order_table = {
            'asap': self._gates_order_asap,
            'alap': self._gates_order_alap,
        }

    def parse(self, qasm, rng=None, *, ordering='ALAP',
              time_start=None, time_end=None):
        return list(self.gen_circuits(
            qasm, rng, ordering=ordering,
            time_start=time_start, time_end=time_end))

    def gen_circuits(self, qasm, rng=None, *, ordering='ALAP',
                     time_start=None, time_end=None):
        rng = ct._ensure_rng(rng)
        if isinstance(qasm, str):
            return self._gen_circuits_fn(qasm, rng, ordering,
                                         time_start, time_end)
        else:
            return self._gen_circuits(qasm, rng, ordering,
                                      time_start, time_end)

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

    def _gen_circuits(self, fp, rng, ordering, time_start, time_end):
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
            # We pass the same rng to avoid correlations between measurements,
            # everything else must be re-initialized
            yield self._parse_circuit(title, source, ordering, rng,
                                      time_start, time_end)

    def _gen_circuits_fn(self, filename, rng, ordering, time_start, time_end):
        with open(filename, 'r') as fp:
            generator = self._gen_circuits(fp, rng, ordering,
                                           time_start, time_end)
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

    @classmethod
    def _gate_spec_to_gate(cls, gate_spec, gate_label, rng):
        """Returns gate with time set to 0. and its duration"""

        duration = gate_spec['duration']
        qubits = gate_spec['qubits']
        if cls._gate_is_ignored(gate_spec):
            return None, 0.
        elif cls._gate_is_measurement(gate_spec):
            seed = rng.randint(1 << 32)
            gate = ct.Measurement(
                qubits[0],
                0.,
                ct.uniform_sampler(np.random.RandomState(seed=seed))
            )
            gate.label = gate_label
        elif cls._gate_is_single_qubit(gate_spec):
            m = np.array(gate_spec['matrix'], dtype=float)
            # TODO: verify if it is not conjugate
            kraus = (m[:, 0] + m[:, 1]*1j).reshape((2, 2))
            gate = ct.SinglePTMGate(
                qubits[0],
                0.,
                ptm.single_kraus_to_ptm(kraus)
            )
            gate.label = gate_label
        elif cls._gate_is_two_qubit(gate_spec):
            m = np.array(gate_spec['matrix'], dtype=float)
            # TODO: verify if it is not conjugate
            kraus = (m[:, 0] + m[:, 1]*1j).reshape((4, 4))
            gate = ct.TwoPTMGate(
                qubits[0],
                qubits[1],
                ptm.double_kraus_to_ptm(kraus),
                0.,
            )
            gate.label = gate_label
        else:
            raise ConfigurationError(
                'Could not identify gate type from gate_spec')

        return gate, duration

    def _parse_circuit(self, title, source, ordering, rng,
                       time_start, time_end):
        gate_specs = [self._instr[line] for line in source]
        qubits = set(chain(*(gs['qubits'] for gs in gate_specs)))
        circuit = ct.Circuit(title)
        for qubit in qubits:
            self._add_qubit(circuit, qubit)

        try:
            gates, tmin, tmax = self._gates_order_table[ordering.lower()](
                qubits, gate_specs, source, rng,
                time_start=time_start, time_end=time_end)
        except KeyError:
            raise RuntimeError('Unknow ordering: {}'.format(ordering))

        for gate, instr in zip(gates, source):
            circuit.add_gate(gate)

        # tmin might be important, tmax is defined by the last gate --
        # idling gates afterwards are useless
        circuit.add_waiting_gates(tmin=tmin, tmax=None)
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

    def _gates_order_alap(self, qubits, gate_specs, gate_labels,
                          rng, time_start, time_end):
        rng = ct._ensure_rng(rng)
        current_times = {qubit: 0. for qubit in qubits}
        gates = []
        for spec, label in zip(reversed(gate_specs),
                               reversed(gate_labels)):
            gate, duration = self._gate_spec_to_gate(spec, label, rng)
            # gate is validated already inside `_gate_spec_to_gate`
            if gate is None:
                continue
            gate_time_end = min((current_times[qubit]
                                 for qubit in spec['qubits']))
            gate_time_start = gate_time_end - duration
            gate.time = 0.5*(gate_time_start + gate_time_end)
            for qubit in spec['qubits']:
                current_times[qubit] = gate_time_start
            gates.append(gate)

        gates = list(reversed(gates))
        time_min = min(current_times.values())
        if time_start is not None and time_end is not None:
            raise RuntimeError('Only start or end time of the circuit '
                               'can be specified')
        elif time_end is not None:
            time_shift = time_end
        else:
            if time_start is None:
                time_start = 0.
            time_shift = time_start - time_min

        if not np.allclose(time_shift, 0.):
            for gate in gates:
                gate.time += time_shift

        return gates, time_min + time_shift, time_shift

    def _gates_order_asap(self, qubits, gate_specs, gate_labels,
                          rng, time_start, time_end):
        rng = ct._ensure_rng(rng)
        current_times = {qubit: 0. for qubit in qubits}
        gates = []
        for spec, label in zip(gate_specs, gate_labels):
            gate, duration = self._gate_spec_to_gate(spec, label, rng)
            # gate is validated already inside `_gate_spec_to_gate`
            if gate is None:
                continue
            gate_time_start = max((current_times[qubit]
                                   for qubit in spec['qubits']))
            gate_time_end = gate_time_start + duration
            gate.time = 0.5*(gate_time_start + gate_time_end)
            for qubit in spec['qubits']:
                current_times[qubit] = gate_time_end
            gates.append(gate)

        time_max = max(current_times.values())
        if time_start is not None and time_end is not None:
            raise RuntimeError('Only start or end time of the circuit '
                               'can be specified')
        elif time_end is None:
            time_shift = time_start or 0.
        else:
            time_shift = time_end - time_max

        if not np.allclose(time_shift, 0.):
            for gate in gates:
                gate.time += time_shift
        return gates, time_shift, time_max + time_shift
