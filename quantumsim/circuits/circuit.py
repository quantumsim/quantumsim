import abc
from copy import copy


class CircuitBase(metaclass=abc.ABCMeta):

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

    @abc.abstractmethod
    def call(self, qubits):
        pass

    def set_qubits(self, qubits):
        circuit = copy(self)
        circuit.qubits = qubits
        return circuit

    def __add__(self, other):
        all_gates = self.gates + other.gates
        all_qubits = self.qubits + tuple(q for q in other.qubits
                                         if q not in self.qubits)
        return Circuit(all_qubits, all_gates)

    def __matmul__(self, other):
        if isinstance(other, CircuitBase):
            raise TypeError("Circuits may only be combined via addition!")
        state = other
        for gate in self.gates:
            state = gate @ state
        return state
        # if isinstance(other, CircuitBase):
        #     return self + other.set_time_start(self.time_end)


class TimedObject(CircuitBase, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def time_start(self):
        pass

    @abc.abstractmethod
    def shift(self, time):
        pass

    @property
    @abc.abstractmethod
    def time_end(self):
        pass

    @property
    @abc.abstractmethod
    def duration(self):
        pass

    def __add__(self, other):
        all_gates = self.gates + other.gates
        all_qubits = self.qubits + tuple(q for q in other.qubits
                                         if q not in self.qubits)
        #NEED TO DO TETRIS!
        raise NotImplementedError
        return TimedCircuit(all_qubits, all_gates)


class Circuit(CircuitBase):
    def __init__(self, qubits, gates):
        self._gates = tuple(gates)
        self._qubits = tuple(qubits)

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

    @classmethod
    def schedule(self, *gate_list):
        raise NotImplementedError


class TimedCircuit(Circuit, TimedObject):

    @property
    def time_start(self):
        return {
            q: min(g.time_start[q] for g in self.gates if q in g.qubits)
            for q in self.qubits}

    def shift(self, time):
        for g in self._gates:
            g.shift(time)

    @property
    def time_end(self):
        return {
            q: max(g.time_start[q] for g in self.gates if q in g.qubits)
            for q in self.qubits}

    @property
    def duration(self):
        return {
            q: max(g.time_start[q] for g in self.gates if q in g.qubits) -
            min(g.time_start[q] for g in self.gates if q in g.qubits)
            for q in self.qubits}


class Gate(CircuitBase):
    def __init__(self, qubits,
                 operations=None,
                 plot_metadata=None):
        self._qubits = (qubits,) if isinstance(qubits, str) else tuple(qubits)
        self.operations = operations or []
        self.plot_metadata = plot_metadata or {}

    @property
    def gates(self):
        return self,

    @property
    def qubits(self):
        return self._qubits

    @property
    def params(self):
        raise NotImplementedError

    @qubits.setter
    def qubits(self, qubits):
        self._qubits = (qubits,) if isinstance(qubits, str) else tuple(qubits)

    def __call__(self, *qubits, **kwargs):

        new_gate = copy(self)
        new_gate.set(*qubits, **kwargs)
        return new_gate

    def set(self, *qubits, **kwargs):

        if len(qubits) != 0:
            self.qubits = qubits

        for key in kwargs:
            if (key not in self.kwargs) and (key not in self.free_kwargs):
                raise KeyError("Parameter '{}' not understood.".format(key))

        for key in kwargs:
            if key in self.kwargs:
                self.kwargs[key] = kwargs[key]
            else:  # key in self.free_kwargs
                self.kwargs[key] = kwargs[key]
                del self.free_kwargs[self.free_kwargs.index(key)]

    def __matmul__(self, other):
        if self.free_kwargs:
            raise NotImplementedError(
                "The following parameters need to be set before"
                " {} may be applied to a qubit register: {}".format(
                    self, self.free_kwargs))

        state = other
        for op in self.operations:
            kwargs_to_send = {
                key: self.kwargs[key]
                for key in op.kwargs
            }
            state = op(**kwargs_to_send) @ state


class TimedGate(Gate, TimedObject):

    def __init__(self, durations, times_start=None, **kwargs):
        '''TimedGate - a gate with a well-defined timing.

        Parameters:
        -----
        duration : dictionary of floats
            the duration of the gate on each of the qubits
        time_start : dictionary of floats or None
            an absolute start time on each of the qubits
        '''
        super().__init__(**kwargs)
        self._times_start = times_start or {q: 0 for q in self.qubits}
        self._durations = durations

    @property
    def time_start(self):
        return self._times_start

    @time_start.setter
    def time_start(self, time_start):
        self._times_start = time_start

    def shift(self, time):
        if self._time_start is None:
            raise ValueError("Absolute time is un-defined")
        for q in self.qubits:
            self._time_end[q] += time
            self._time_start[q] += time

    @property
    def time_end(self):
        return {q: self.time_start[q] + self.duration[q]
                for q in self.qubits}

    @property
    def duration(self):
        return self._duration
