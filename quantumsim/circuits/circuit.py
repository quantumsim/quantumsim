import abc


class CircuitBase(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def time_start(self):
        pass

    @property
    @abc.abstractmethod
    def time_end(self):
        pass

    @property
    @abc.abstractmethod
    def duration(self):
        pass

    @property
    @abc.abstractmethod
    def gates(self):
        pass

    @property
    @abc.abstractmethod
    def qubits(self):
        pass

    def __add__(self, other):
        all_gates = sorted(self.gates + other.gates, key=lambda g: g.time_start)
        all_qubits = self.qubits + tuple(q for q in other.qubits
                                         if q not in self.qubits)
        return Circuit(all_qubits, all_gates)


class Circuit(CircuitBase):
    def __init__(self, qubits, gates):
        self._gates = gates
        self._qubits = qubits

    @property
    def time_start(self):
        return min(g.time_start for g in self.gates)

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


class Gate(CircuitBase):
    def __init__(self, qubits, duration=0., time_start=0., operations=None,
                 plot_metadata=None):
        self._qubits = (qubits,) if isinstance(qubits, str) else qubits
        self._time_start = time_start
        self._duration = duration
        self.operations = operations
        self.plot_metadata = plot_metadata or {}

    @property
    def time_start(self):
        return self._time_start

    @property
    def time_end(self):
        return self.time_start + self.duration

    @property
    def duration(self):
        return self._duration

    @property
    def gates(self):
        return [self]

    @property
    def qubits(self):
        return self._qubits
