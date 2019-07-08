import abc
from copy import copy


class CircuitBase(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def time_start(self):
        pass

    @time_start.setter
    @abc.abstractmethod
    def time_start(self, time):
        pass

    @property
    @abc.abstractmethod
    def time_end(self):
        pass

    @property
    @abc.abstractmethod
    def duration(self):
        pass

    def set_time_start(self, time_start):
        g = copy(self)
        g.time_start = time_start
        return g

    @property
    @abc.abstractmethod
    def gates(self):
        pass

    @property
    @abc.abstractmethod
    def qubits(self):
        pass

    @qubits.setter
    @abc.abstractmethod
    def qubits(self, qubits):
        pass

    def set_qubits(self, qubits):
        circuit = copy(self)
        circuit.qubits = qubits
        return circuit

    def __add__(self, other):
        all_gates = sorted(self.gates + other.gates, key=lambda g: g.time_start)
        all_qubits = self.qubits + tuple(q for q in other.qubits
                                         if q not in self.qubits)
        return Circuit(all_qubits, all_gates)


class Circuit(CircuitBase):
    def __init__(self, qubits, gates):
        self._gates = tuple(gates)
        self._qubits = tuple(qubits)

    @property
    def time_start(self):
        return min(g.time_start for g in self.gates)

    @time_start.setter
    def time_start(self, time_start):
        self._gates = tuple(copy(g) for g in self._gates)
        time_shift = time_start - self.time_start
        for g in self._gates:
            g.time_start += time_shift

    @property
    def time_end(self):
        return max(g.time_end for g in self.gates)

    @property
    def duration(self):
        return self.time_end - self.time_start

    @property
    def gates(self):
        return self._gates

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, qubits):
        old_new_mapping = {old: new for old, new in zip(self._qubits, qubits)}
        self._qubits = (qubits,) if isinstance(qubits, str) else qubits
        self._gates = tuple(copy(g) for g in self._gates)
        for g in self._gates:
            g._qubits = tuple(old_new_mapping[q] for q in g._qubits)


class Gate(CircuitBase):
    def __init__(self, qubits, duration=0., time_start=0., operations=None,
                 plot_metadata=None):
        self._qubits = (qubits,) if isinstance(qubits, str) else tuple(qubits)
        self._time_start = time_start
        self._duration = duration
        self.operations = operations
        self.plot_metadata = plot_metadata or {}

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, time_start):
        self._time_start = time_start

    @property
    def time_end(self):
        return self.time_start + self.duration

    @property
    def duration(self):
        return self._duration

    def set_time_start(self, time_start):
        g = copy(self)
        g.time_start = time_start
        return g

    @property
    def gates(self):
        return self,

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, qubits):
        self._qubits = (qubits,) if isinstance(qubits, str) else tuple(qubits)
