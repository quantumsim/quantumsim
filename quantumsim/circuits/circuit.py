import abc
import inspect
import re
from copy import copy

from quantumsim import Operation


class CircuitBase(metaclass=abc.ABCMeta):
    _valid_identifier_re = re.compile('[a-zA-Z_][a-zA-Z0-9_]*')

    @abc.abstractmethod
    def operation(self, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def qubits(self):
        pass

    @abc.abstractmethod
    def __call__(self, qubits):
        pass

    def __add__(self, other):
        all_gates = self.gates + other.gates
        all_qubits = self.qubits + tuple(q for q in other.qubits
                                         if q not in self.qubits)
        return Circuit(all_qubits, all_gates)

    def __matmul__(self, other):
        if isinstance(other, CircuitBase):
            raise TypeError("Circuits may only be combined via addition!")
        for gate in self.gates:
            state = gate @ state
        return state
        # if isinstance(other, CircuitBase):
        #     return self + other.set_time_start(self.time_end)


class TimedCircuitBase(CircuitBase, metaclass=abc.ABCMeta):

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


class TimedCircuit(Circuit, TimedCircuitBase):

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
    def __init__(self, qubits, operation, plot_metadata=None):
        """A gate without notion of timing.

        Parameters
        ----------
        qubits : str or list of str
            Names of the involved qubits
        operation : quantumsim.Operation or function
            Operation, that corresponds to this gate, or a function,
            that takes a certain number of arguments (gate parameters) and
            returns an operation.
        plot_metadata : None or dict
            Metadata, that describes how to represent a gate on a plot.
            TODO: link documentation, when plotting is ready.
        """
        self._qubits = (qubits,) if isinstance(qubits, str) else tuple(qubits)
        if isinstance(operation, Operation):
            self._operation_func = lambda: operation
            self._params_real = tuple()
            self._params = set()
        elif callable(operation):
            self._operation_func = operation
            argspec = inspect.getfullargspec(operation)
            if argspec.varargs is not None:
                raise ValueError(
                    "`operation` function can't accept free arguments.")
            if argspec.varkw is not None:
                raise ValueError(
                    "`operation` function can't accept free keyword arguments.")
            self._params_real = tuple(argspec.args)
            self._params = set(self._params_real)
        else:
            raise ValueError('`operation` argument must be either Operation, '
                             'or a function, that returns Operation.')
        self._params_set = {}
        self._params_subs = {}
        self.plot_metadata = plot_metadata or {}

    def operation(self, **kwargs):
        kwargs.update(self._params_set)  # set parameters take priority
        try:
            for name, real_name in self._params_subs.items():
                kwargs[real_name] = kwargs.pop(name)
            args = tuple(kwargs[name] for name in self._params_real)
        except KeyError as err:
            raise RuntimeError(
                "Can't construct an operation for gate {}, "
                "since parameter \"{}\" is not provided."
                .format(repr(self), err.args[0]))
        op = self._operation_func(*args)
        if not isinstance(op, Operation):
            raise RuntimeError(
                'Invalid operation function was provided for the gate {} '
                'during its creation: it must return quantumsim.Operation. '
                'See quantumsim.Gate documentation for more information.'
                .format(repr(self)))
        if not op.num_qubits == len(self.qubits):
            raise RuntimeError(
                'Invalid operation function was provided for the gate {} '
                'during its creation: its number of qubits does not match '
                'one of the gate. '
                'See quantumsim.Gate documentation for more information.'
                .format(repr(self)))
        return op

    @property
    def gates(self):
        return self,

    @property
    def qubits(self):
        return self._qubits

    @property
    def params(self):
        return self._params

    def _set_param(self, name, value):
        if name not in self._params:
            return
        real_name = self._params_subs.pop(name, name)
        if isinstance(value, str):
            if self._valid_identifier_re.match(value) is None:
                raise ValueError("\"{}\" is not a valid Python "
                                 "identifier.".format(value))
            self._params_subs[value] = real_name
            self._params.add(value)
        else:
            self._params_set[real_name] = value
        self._params.remove(name)

    def set(self, **kwargs):
        for item in kwargs.items():
            self._set_param(*item)

    def __call__(self, *qubits, **kwargs):
        new_gate = copy(self)
        new_gate.set(*qubits, **kwargs)
        return new_gate

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


class TimedGate(Gate, TimedCircuitBase):

    @property
    def params(self):
        pass

    def __init__(self, durations, times_start=None, **kwargs):
        """TimedGate - a gate with a well-defined timing.

        Parameters:
        -----
        duration : dictionary of floats
            the duration of the gate on each of the qubits
        time_start : dictionary of floats or None
            an absolute start time on each of the qubits
        """
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
