import abc
import inspect
import re
from abc import ABCMeta
from contextlib import contextmanager
from copy import copy
from itertools import chain

from .. import Operation

parameter_collisions_allowed = False

# TODO: implement scheduling


@contextmanager
def allow_parameter_collisions():
    global parameter_collisions_allowed
    parameter_collisions_allowed = True
    yield
    parameter_collisions_allowed = False


class CircuitBase(metaclass=abc.ABCMeta):
    _valid_identifier_re = re.compile('[a-zA-Z_][a-zA-Z0-9_]*')

    @abc.abstractmethod
    def __copy__(self):
        pass

    @abc.abstractmethod
    def operation(self, **kwargs):
        """Convert a gate to a raw operation."""
        pass

    @property
    @abc.abstractmethod
    def qubits(self):
        """Qubit names, associated with this circuit."""
        pass

    @property
    @abc.abstractmethod
    def params(self):
        """Return set of parameters, accepted by this circuit."""
        pass

    @abc.abstractmethod
    def set(self, **kwargs):
        """Either substitute a circuit parameter with a value, or rename it.

        Arguments to this function is a mapping of old parameter name to
        either its name, or a value. If type of a value provided is
        :class:`str`, it is interpreted as a new parameter name, else as a
        value for this parameter.
        """
        pass

    def __call__(self, **kwargs):
        """Convenience method to copy a circuit with parameters updated. See
        :func:`CircuitBase.set` for a description.
        """
        copy_ = copy(self)
        copy_.set(**kwargs)
        return copy_


class Gate(CircuitBase, metaclass=ABCMeta):
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

    def __call__(self, **kwargs):
        new_gate = copy(self)
        new_gate.set(**kwargs)
        return new_gate


class TimeAgnostic(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def qubits(self):
        pass

    @property
    @abc.abstractmethod
    def gates(self):
        pass

    @property
    @abc.abstractmethod
    def params(self):
        pass

    def __add__(self, other):
        """
        Merge two circuits, locating second one after first.

        Parameters
        ----------
        other : TimeAgnostic
            Another circuit

        Returns
        -------
        TimeAgnosticCircuit
            A merged circuit.
        """
        global parameter_collisions_allowed
        if not parameter_collisions_allowed:
            common_params = self.params.intersection(other.params)
            if len(common_params) > 0:
                raise RuntimeError(
                    "The following free parameters are common for the circuits "
                    "being added, which blocks them from being set "
                    "separately later:\n"
                    "   {}\n"
                    "Rename these parameters in one of the circuits, or use "
                    "`quantumsim.circuits.allow_parameter_collisions` "
                    "context manager, if this is intended behaviour."
                    .format(", ".join(common_params))
                )
        all_gates = self.gates + other.gates
        all_qubits = self.qubits + tuple(q for q in other.qubits
                                         if q not in self.qubits)
        return TimeAgnosticCircuit(all_qubits, all_gates)


class TimeAware(metaclass=abc.ABCMeta):

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
        pass


class Circuit(CircuitBase):
    def __init__(self, qubits, gates):
        self._gates = tuple(gates)
        self._qubits = tuple(qubits)
        self._params_cache = None

    @property
    def qubits(self):
        return self._qubits

    @property
    def gates(self):
        return self._gates

    @property
    def params(self):
        if self._params_cache is None:
            self._params_cache = set(chain(*(g.params for g in self._gates)))
        return self._params_cache

    def operation(self, **kwargs):
        operations = []
        for gate in self._gates:
            qubit_indices = tuple(self._qubits.index(qubit) for qubit in
                                  gate.qubits)
            operations.append(gate.operation(**kwargs).at(*qubit_indices))
        return Operation.from_sequence(operations)

    def set(self, **kwargs):
        for gate in self._gates:
            gate.set(**kwargs)
        self._params_cache = None


class TimeAgnosticGate(Gate, TimeAgnostic):
    def __copy__(self):
        copy_ = self.__class__(
            self._qubits, self._operation_func, self.plot_metadata)
        copy_._params_set = copy(self._params_set)
        copy_._params_subs = copy(self._params_subs)
        return copy_


class TimeAgnosticCircuit(Circuit, TimeAgnostic):
    def __copy__(self):
        # Need a shallow copy of all included gates
        copy_ = self.__class__(self._qubits, (copy(g) for g in self._gates))
        copy_._params_cache = self._params_cache
        return copy_


class TimeAwareGate(Gate, TimeAware):

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

    def __copy__(self):
        raise NotImplementedError

    @property
    def time_start(self):
        return self._times_start

    @time_start.setter
    def time_start(self, time_start):
        self._times_start = time_start

    def shift(self, time):
        raise NotImplementedError

    @property
    def time_end(self):
        return {q: self.time_start[q] + self.duration[q]
                for q in self.qubits}

    @property
    def duration(self):
        raise NotImplementedError


class TimeAwareCircuit(Circuit, TimeAware):
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

    def __copy__(self):
        raise NotImplementedError
