import abc
from contextlib import contextmanager
from copy import copy
from itertools import chain

from ..operations import Operation, ParametrizedOperation

param_repeat_allowed = False

# TODO: implement scheduling


@contextmanager
def allow_param_repeat():
    global param_repeat_allowed
    param_repeat_allowed = True
    yield
    param_repeat_allowed = False


class CircuitBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __copy__(self):
        pass

    @property
    @abc.abstractmethod
    def operation(self):
        """Convert a gate to a raw operation."""
        pass

    @property
    @abc.abstractmethod
    def qubits(self):
        """Qubit names, associated with this circuit."""
        pass

    @property
    @abc.abstractmethod
    def gates(self):
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


class Gate(CircuitBase, metaclass=abc.ABCMeta):
    def __init__(self, qubits, operation, plot_metadata=None):
        """A gate without notion of timing.

        Parameters
        ----------
        qubits : str or list of str
            Names of the involved qubits
        operation : quantumsim.Operation
            Operation, that corresponds to this gate.
        plot_metadata : None or dict
            Metadata, that describes how to represent a gate on a plot.
            TODO: link documentation, when plotting is ready.
        """
        assert isinstance(operation, Operation)
        self._qubits = (qubits,) if isinstance(qubits, str) else tuple(qubits)
        self._operation = operation
        if len(self._qubits) != operation.num_qubits:
            raise ValueError('Number of qubits in operation does not match '
                             'one in `qubits`.')
        self.plot_metadata = plot_metadata or {}

    @property
    def operation(self):
        return self._operation

    @property
    def gates(self):
        return self,

    @property
    def qubits(self):
        return self._qubits

    @property
    def params(self):
        if isinstance(self._operation, ParametrizedOperation):
            return self._operation.params
        else:
            return set()

    def set(self, **kwargs):
        if isinstance(self._operation, ParametrizedOperation):
            self._operation = self._operation.substitute(**kwargs)

    def __call__(self, **kwargs):
        new_gate = copy(self)
        new_gate.set(**kwargs)
        return new_gate


class CircuitAddMixin(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def __add__(self, other):
        global param_repeat_allowed
        if not param_repeat_allowed:
            common_params = self.params.intersection(other.params)
            if len(common_params) > 0:
                raise RuntimeError(
                    "The following free parameters are common for the circuits "
                    "being added, which blocks them from being set "
                    "separately later:\n"
                    "   {}\n"
                    "Rename these parameters in one of the circuits, or use "
                    "`quantumsim.circuits.allow_param_repeat` "
                    "context manager, if this is intended behaviour."
                    .format(", ".join(common_params)))


class TimeAgnostic(CircuitAddMixin, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def qubits(self):
        pass

    @property
    @abc.abstractmethod
    def gates(self):
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
        super().__add__(other)
        all_gates = self.gates + other.gates
        all_qubits = self.qubits + tuple(q for q in other.qubits
                                         if q not in self.qubits)
        return TimeAgnosticCircuit(all_qubits, all_gates)


class TimeAware(CircuitAddMixin, metaclass=abc.ABCMeta):
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

    @time_end.setter
    @abc.abstractmethod
    def time_end(self, time):
        pass

    @property
    @abc.abstractmethod
    def duration(self):
        pass

    @property
    @abc.abstractmethod
    def qubits(self):
        pass

    @property
    @abc.abstractmethod
    def gates(self):
        pass

    def shift(self, *, time_start=None, time_end=None):
        """

        Parameters
        ----------
        time_start : float or None
        time_end : float or Nont

        Returns
        -------
        TimeAware
        """
        if time_start is not None and time_end is not None:
            raise ValueError('Only one argument is accepted.')
        copy_ = copy(self)
        if time_start is not None:
            copy_.time_start = time_start
        elif time_end is not None:
            copy_.time_end = time_end
        else:
            raise ValueError('Specify time_start or time_end')
        return copy_

    def _qubit_time_start(self, qubit):
        for gate in self.gates:
            if qubit in gate.qubits:
                return gate.time_start

    def _qubit_time_end(self, qubit):
        for gate in reversed(self.gates):
            if qubit in gate.qubits:
                return gate.time_end

    def __add__(self, other):
        """

        Parameters
        ----------
        other : TimeAware
            Another circuit.

        Returns
        -------
        TimeAwareCircuit
        """
        super().__add__(other)
        shared_qubits = set(self.qubits).intersection(other.qubits)
        if len(shared_qubits) > 0:
            other_shifted = other.shift(time_start=max(
                (self._qubit_time_end(q) - other._qubit_time_start(q)
                 for q in shared_qubits)) + 2*other.time_start)
        else:
            other_shifted = copy(other)
        qubits = tuple(chain(self.qubits,
                             (q for q in other.qubits if q not in self.qubits)))
        gates = tuple(chain((copy(g) for g in self.gates),
                            other_shifted.gates))
        return TimeAwareCircuit(qubits, gates)


class Circuit(CircuitBase, metaclass=abc.ABCMeta):
    def __init__(self, qubits, gates):
        self._gates = tuple(gates)
        self._qubits = tuple(qubits)
        self._params_cache = None
        self._operation = None

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

    @property
    def operation(self):
        operations = []
        for gate in self._gates:
            qubit_indices = tuple(self._qubits.index(qubit) for qubit in
                                  gate.qubits)
            operations.append(gate.operation.at(*qubit_indices))
        return Operation.from_sequence(operations)

    def set(self, **kwargs):
        for gate in self._gates:
            gate.set(**kwargs)
        self._params_cache = None


class TimeAgnosticGate(Gate, TimeAgnostic):
    def __copy__(self):
        return self.__class__(
            self._qubits, copy(self._operation), self.plot_metadata)


class TimeAgnosticCircuit(Circuit, TimeAgnostic):
    def __copy__(self):
        # Need a shallow copy of all included gates
        copy_ = self.__class__(self._qubits, (copy(g) for g in self._gates))
        copy_._params_cache = self._params_cache
        return copy_


class TimeAwareGate(Gate, TimeAware):

    def __init__(self, qubits, operation, duration=0.,
                 time_start=0., plot_metadata=None):
        """TimedGate - a gate with a well-defined timing.

        Parameters:
        -----
        duration : dictionary of floats
            the duration of the gate on each of the qubits
        time_start : dictionary of floats or None
            an absolute start time on each of the qubits
        """
        super().__init__(qubits, operation, plot_metadata)
        self._duration = duration
        self._time_start = time_start

    def __copy__(self):
        return self.__class__(
            self._qubits, copy(self._operation),
            self._duration, self._time_start, self.plot_metadata)

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, time):
        self._time_start = time

    @property
    def time_end(self):
        return self._time_start + self._duration

    @time_end.setter
    def time_end(self, time):
        self._time_start = time - self._duration

    @property
    def duration(self):
        return self._duration


class TimeAwareCircuit(Circuit, TimeAware):
    def __init__(self, qubits, gates):
        super().__init__(qubits, gates)
        self._time_start = min((g.time_start for g in self._gates))
        self._time_end = max((g.time_end for g in self._gates))

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, time):
        shift = time - self._time_start
        for g in self._gates:
            g.time_start += shift
        self._time_start += shift
        self._time_end += shift

    @property
    def time_end(self):
        return self._time_end

    @time_end.setter
    def time_end(self, time):
        shift = time - self.time_end
        self.time_start += shift

    @property
    def duration(self):
        return self._time_end - self._time_start

    def __copy__(self):
        # Need a shallow copy of all included gates
        copy_ = self.__class__(self._qubits, (copy(g) for g in self._gates))
        copy_._params_cache = self._params_cache
        return copy_


class FinalizedCircuit:
    """
    Parameters
    ----------
    Circuit : CircuitBase
        A circuit to finalize
    """
    def __init__(self, circuit, *, bases_in=None):
        if circuit:
            self.operation = circuit.operation.compile(bases_in=bases_in)

            self.qubits = copy(circuit.qubits)
            if hasattr(self.operation, 'operations'):
                self._params = set().union((op.params for op in filter(
                    lambda op: isinstance(op, ParametrizedOperation),
                    self.operation)))
            elif hasattr(self.operation, 'params'):
                self._params = self.operation.params
            else:
                self._params = set()

    def __call__(self, **params):
        if len(self._params) > 0:
            unset_params = self._params - params.keys()
            if len(unset_params) != 0:
                raise KeyError(*unset_params)
            out = FinalizedCircuit(None)
            out.operation = self.operation.substitute(**params)\
                .compile(optimize=False)
            out._params = set()
            return out
        else:
            return self

    def __matmul__(self, state):
        """

        Parameters
        ----------
        state : quantumsim.State
        """
        if len(self._params) != 0:
            raise KeyError(*self._params)
        try:
            indices = [state.qubits.index(q) for q in self.qubits]
        except ValueError as ex:
            raise ValueError(
                'Qubit {} is not present in the state'
                .format(ex.args[0].split()[0]))
        self.operation(state.pauli_vector, *indices)
