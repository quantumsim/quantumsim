import abc
import numpy as np
from collections import defaultdict
from more_itertools import pairwise

from quantumsim import Operation
from quantumsim.circuits import TimeAgnosticGate, TimeAwareGate, \
    FinalizedCircuit, TimeAwareCircuit
from ..operations import Placeholder
from .. import bases


from ..operations.operation import IndexedOperation


class WaitPlaceholder(Placeholder):
    def __init__(self, duration, dim):
        super().__init__((bases.general(dim),))
        self.duration = duration


class Model(metaclass=abc.ABCMeta):
    """
    Parameters
    ----------
    setup : quantumsim.Setup
    seed : int
        Seed for initializing an internal random number generator.
    """
    def __init__(self, setup, seed=None):
        self._setup = setup
        self.rng = np.random.RandomState(seed)

    def wait(self, qubit, duration):
        return WaitPlaceholder(duration, self.dim).at(qubit)

    def waiting_gate(self, qubit, duration):
        return TimeAwareGate(qubit, WaitPlaceholder(duration, self.dim),
                             duration,
                             plot_metadata={'style': 'marker', 'label': 'x'})

    def p(self, param, *qubits):
        return self._setup.param(param, *qubits)

    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @staticmethod
    def _normalize_operation(op, qubits):
        """

        Parameters
        ----------
        op : Operation or IndexedOperation
        qubits : str or list of str

        Returns
        -------
        IndexedOperation
        """
        if isinstance(op, Operation):
            if len(qubits) > 1:
                raise ValueError(
                    "Can't construct an operation from a sequence: "
                    "all operations in multi-qubit gate must provide "
                    "indices using Operation.at() method.")
            return op.at(0)
        elif isinstance(op, IndexedOperation):
            op, ix = op
            ix = tuple(qubits.index(qubit) for qubit in ix)
            return op.at(*ix)
        else:
            raise ValueError(
                "`op` can be only Operation or IndexedOperation, got {}"
                .format(type(op)))

    @staticmethod
    def gate(duration=None, plot_metadata=None):
        def gate_decorator(func):
            def make_operation(self, *qubits):
                sequence = func(self, *qubits)
                sequence = sequence if (isinstance(sequence, tuple) or
                                        isinstance(sequence, list)) else \
                    (sequence,)
                sequence = [self._normalize_operation(op, qubits) for op
                            in sequence]
                return Operation.from_sequence(sequence, qubits)

            if duration is None:
                def wrapper(self, *qubits, **params):
                    return TimeAgnosticGate(
                        qubits, make_operation(self, *qubits),
                        plot_metadata)(**params)
            else:
                def wrapper(self, *qubits, **params):
                    if callable(duration):
                        _duration = duration(*qubits, self._setup)
                    elif isinstance(duration, str):
                        _duration = self.p(duration, *qubits)
                    else:
                        _duration = duration
                    return TimeAwareGate(
                        qubits, make_operation(self, *qubits),
                        _duration, 0., plot_metadata)(**params)
            wrapper.__name__ = func.__name__
            return wrapper
        return gate_decorator

    def add_waiting_gates(self, circuit):
        """Insert missing waiting placeholders and return finalized circuit.

        Parameters
        ----------
        circuit : quantumsim.circuits.TimeAware

        Returns
        -------
        quantumsim.circuits.Circuit
        """
        gates_dict = defaultdict(list)
        for gate in circuit.gates:
            for qubit in gate.qubits:
                gates_dict[qubit].append(gate)
        time_start = circuit.time_start
        time_end = circuit.time_end
        margin = 1e-1
        waiting_gates = []

        for qubit, gates in gates_dict.items():
            duration = gates[0].time_start - time_start
            if duration > margin:
                waiting_gates.append(
                    self.waiting_gate(qubit, duration)
                        .shift(time_start=time_start))
            duration = time_end - gates[-1].time_end
            if duration > margin:
                waiting_gates.append(
                    self.waiting_gate(qubit, duration)
                        .shift(time_end=time_end))
            for gate1, gate2 in pairwise(gates):
                duration = gate2.time_start - gate1.time_end
                if duration > margin:
                    waiting_gates.append(self.waiting_gate(qubit, duration)
                                         .shift(time_start=gate1.time_end))
        gates = sorted(circuit.gates + tuple(waiting_gates),
                       key=lambda g: g.time_start)
        return TimeAwareCircuit(circuit.qubits, gates)

    def finalize(self, circuit, bases_in=None):
        """
        This function is aimed to perform post-processing operations on a
        circuit (such as, for example, adding waiting gates) and returns a
        finalized version of this circuit.

        Parameters
        ----------
        circuit: quantumsim.circuits.CircuitBase
            A circuit
        bases_in: tuple of quantumsim.PauliBasis

        Returns
        -------
        quantumsim.circuits.FinalizedCircuit
            A post-processed and finalized version of the circuit.
        """
        return FinalizedCircuit(circuit.operation, circuit.qubits,
                                bases_in=bases_in)
