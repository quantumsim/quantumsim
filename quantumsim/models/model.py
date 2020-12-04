import abc
from collections import defaultdict
from copy import copy, deepcopy
from itertools import tee

from ..circuits.circuit import GatePlaceholder, Circuit, Box


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class WaitingGate(GatePlaceholder):
    def __init__(self, qubit, duration, dim_hilbert, time_start=0, plot_metadata=None,
                 **metadata):
        super().__init__(qubits=[qubit],
                         dim_hilbert=dim_hilbert,
                         duration=duration,
                         time_start=time_start,
                         plot_metadata=plot_metadata or {
                             "style": "box",
                             "label": r"$\mathcal{{W}}$"
                         })
        self.metadata = metadata

    def __copy__(self):
        other = WaitingGate(self._qubits[0], self.duration, self.dim_hilbert,
                            self._time_start)
        other.metadata = deepcopy(self.metadata)
        return other

    def split(self, time):
        if time < self.time_start or time > self.time_end:
            raise ValueError("time must be between gate's time_start and time_end")
        return (WaitingGate(self._qubits[0],
                            time - self.time_start,
                            self.dim_hilbert,
                            self.time_start,
                            deepcopy(self.plot_metadata),
                            **self.metadata),
                WaitingGate(self._qubits[0],
                            self.time_end - time,
                            self.dim_hilbert,
                            time,
                            deepcopy(self.plot_metadata),
                            **self.metadata))


class Model(metaclass=abc.ABCMeta):
    """
    Parameters
    ----------
    setup : quantumsim.Setup
    """
    def __init__(self, setup, dim=None):
        self._setup = setup
        self._dim = dim

    def wait(self, qubit, duration):
        return WaitingGate(qubit, duration, self.dim)

    def p(self, param, *qubits):
        return self._setup.param(param, *qubits)

    @property
    def dim(self):
        return self._dim

    @staticmethod
    def gate(duration=0, plot_metadata=None, repr_=None):
        def gate_decorator(func):
            def wrapper(self, *qubits, **params):
                if callable(duration):
                    _duration = duration(*qubits, self._setup)
                elif isinstance(duration, str):
                    _duration = self.p(duration, *qubits)
                else:
                    _duration = duration
                circuit = func(self, *qubits)
                circuit.set(**params)
                return Box(qubits, circuit.gates, plot_metadata, repr_)

            wrapper.__name__ = func.__name__
            return wrapper

        return gate_decorator

    def add_waiting_gates(self, circuit):
        """Insert missing waiting placeholders.

        Parameters
        ----------
        circuit : quantumsim.circuits.Circuit

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
                    self.wait(qubit, duration).shift(time_start=time_start))
            duration = time_end - gates[-1].time_end
            if duration > margin:
                waiting_gates.append(
                    self.wait(qubit, duration).shift(time_end=time_end))
            for gate1, gate2 in pairwise(gates):
                duration = gate2.time_start - gate1.time_end
                if duration > margin:
                    waiting_gates.append(self.wait(qubit, duration)
                                             .shift(time_start=gate1.time_end))
        gates = sorted(circuit.gates + waiting_gates,
                       key=lambda g: g.time_start)
        return Circuit(circuit.qubits, gates)

    def finalize(self, circuit, bases_in=None, qubits=None):
        """
        This function is aimed to perform post-processing operations on a
        circuit (such as, for example, adding waiting gates) and returns a
        finalized version of this circuit.

        Parameters
        ----------
        circuit: quantumsim.circuits.CircuitBase
            A circuit
        qubits: list of hashable
            Qubits of the circuit, to set qubit order.
        bases_in: tuple of quantumsim.PauliBasis

        Returns
        -------
        quantumsim.circuits.FinalizedCircuit
            A post-processed and finalized version of the circuit.
        """
        return circuit.finalize(bases_in=bases_in, qubits=qubits)
