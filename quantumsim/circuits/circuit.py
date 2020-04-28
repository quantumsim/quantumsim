import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import copy, deepcopy
from itertools import chain

import numpy as np
from pytools import flatten
from sympy import symbols, sympify

from ..operations.operation import Operation, ParametrizedOperation
from ..operations.operation import _Chain

PARAM_REPEAT_ALLOWED = False


@contextmanager
def allow_param_repeat():
    """Context manager to allow using the same named parameter in
    different gates of a circuit."""
    global PARAM_REPEAT_ALLOWED  # pylint: disable=global-statement
    PARAM_REPEAT_ALLOWED = True
    yield
    PARAM_REPEAT_ALLOWED = False


def sympy_to_native(expr):
    try:
        # Must be complex for sure
        c = complex(expr)
    except Exception as ex:
        raise RuntimeError(
            "Could not convert sympy symbol to native type. "
            "It may be due to misinterpretation of some symbols by sympy. "
            "Try to use sympy expressions as gate parameters' values "
            "explicitly."
        ) from ex
    try:
        f = float(expr)
    except TypeError:
        f = np.nan
    if not np.allclose(c, f):
        return c
    try:
        c = int(expr)
    except TypeError:
        c = np.nan
    i = int(expr)
    if not np.allclose(i, f):
        return f
    return i


class GateSetMixin(ABC):
    """Abstract class, that defines an interface for all gates manipulation."""

    @abstractmethod
    def __copy__(self):
        pass

    @property
    @abstractmethod
    def qubits(self):
        """Qubit names, associated with this circuit."""
        pass

    @property
    @abstractmethod
    def gates(self):
        """Gates (logical units) of this circuit."""
        pass

    def operations(self):
        """Generator of operations (Mathematical units) of this circuit."""
        pass

    @property
    @abstractmethod
    def free_parameters(self):
        """Return set of parameters, accepted by this circuit."""
        pass

    @abstractmethod
    def set(self, **kwargs):
        """Either substitute a circuit parameter with a value, or rename it.

        Arguments to this function is a mapping of old parameter name to
        either its name, or a value. If type of a value provided is
        :class:`str`, it is interpreted as a new parameter name, else as a
        value for this parameter.
        """
        pass

    def shift(self, time_start=None, time_end=None):
        """

        Parameters
        ----------
        time_start : float or None
        time_end : float or None

        Returns
        -------
        GateSetMixin
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

    def __add__(self, other):
        """

        Parameters
        ----------
        other : GateSetMixin
            Another circuit.

        Returns
        -------
        Circuit
        """
        global PARAM_REPEAT_ALLOWED  # pylint: disable=global-statement
        if not PARAM_REPEAT_ALLOWED:
            common_params = self.free_parameters.intersection(
                other.free_parameters)
            if len(common_params) > 0:
                raise RuntimeError(
                    "The following free parameters are common for the circuits "
                    "being added, which blocks them from being set "
                    "separately later:\n"
                    "   {}\n"
                    "Rename these parameters in one of the circuits, or use "
                    "`quantumsim.circuits.allow_param_repeat` "
                    "context manager, if this is intended behaviour."
                    .format(", ".join((str(p) for p in common_params))))
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
        return Circuit(qubits, gates)

    def __call__(self, **kwargs):
        """Convenience method to copy a circuit with parameters updated. See
        :func:`CircuitBase.set` for a description.
        """
        copy_ = copy(self)
        copy_.set(**kwargs)
        return copy_

    def finalize(self, preprocessors=None, bases_in=None):
        """
        Returns an optimized version of the circuit, that can be used to
        apply to the state.

        It will compile together all gates, that do not have params.
        `preprocessors` can be used to adjust the operation before compiling,
        for example, to instantiate custom placeholders, defined by the model.

        Parameters
        ----------
        bases_in : tuple of quantumsim.bases.PauliBasis
        preprocessors: list of functions
            Functions, that take an :class:`Operation` as input and return
            another :class:`Operation`.

        Returns
        -------
        FinalizedCircuit
            Finalized version of this circuit
        """
        return FinalizedCircuit(self.qubits, self.operations(), bases_in=bases_in)

    def _qubit_time_start(self, qubit):
        for gate in self.gates:
            if qubit in gate.qubits:
                return gate.time_start

    def _qubit_time_end(self, qubit):
        for gate in reversed(self.gates):
            if qubit in gate.qubits:
                return gate.time_end


class CircuitUnitMixin(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_metadata = {}
        self._repr = "gate"

    @property
    @abstractmethod
    def params(self):
        """Return a mapping between parameters, used in the definition of this unit,
        and values (numbers, Sympy symbols, etc.), used for its visualizing.
        """
        # FIXME: probably parameters should be handled on a Mixin level,
        # and not on a Gate level.
        pass

    def __repr__(self):
        return self._repr.format(**self.params) + " @ (" + ", ".join(self.qubits) + ")"

    def __str__(self):
        return self._repr.format(**self.params) + " @ (" + ", ".join(self.qubits) + ")"


class Gate(GateSetMixin, CircuitUnitMixin):
    _valid_identifier_re = re.compile('[a-zA-Z_][a-zA-Z0-9_]*')
    _sympify_locals = {
        'beta': symbols('beta'),
        'gamma': symbols('gamma'),
    }

    def __init__(self, qubits, dim_hilbert, operation, duration=0,
                 time_start=0., plot_metadata=None, repr_=None):
        """Gate

        Parameters
        ----------
        qubits : str or list of str
            Names of the involved qubits
        dim_hilbert : int
            Hilbert dimensionality of the correspondent operations
        operation : quantumsim.Operation
            Operation, that corresponds to this gate.
        plot_metadata : None or dict
            Metadata, that describes how to represent a gate on a plot.
            TODO: link documentation, when plotting is ready.
        """
        super(Gate, self).__init__()
        self.dim_hilbert = dim_hilbert
        if isinstance(qubits, str):
            self._qubits = (qubits,)
        elif hasattr(qubits, '__iter__'):
            self._qubits = tuple(qubits)
            for q in self._qubits:
                if not isinstance(q, str):
                    raise ValueError('qubits must be a string or list of '
                                     'strings, got elements of type {}'
                                     .format(type(q)))
        else:
            raise ValueError('qubits must be a string or list of '
                             'strings, got type {}'.format(type(qubits)))

        self._operation = operation
        if len(self._qubits) != operation.num_qubits:
            raise ValueError('Number of qubits in operation does not match '
                             'one in `qubits`.')
        if plot_metadata:
            self.plot_metadata = plot_metadata
        if repr_:
            self._repr = repr_
        self._params = {
            param: symbols(param) for param in
            self._operation_params(self._operation)}
        self._duration = duration
        self._time_start = time_start

    def __copy__(self):
        other = Gate(
            self._qubits, self.dim_hilbert, copy(self._operation),
            self._duration, self._time_start, self.plot_metadata, self._repr)
        other._params = copy(self._params)
        return other

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

    def operation_sympified(self):
        assert not isinstance(self._operation, _Chain), "need to refactor, sorry"
        if isinstance(self._operation, ParametrizedOperation):
            new_op = copy(self._operation)
            new_op.params = tuple(
                self._params[p] if p in self._params.keys() else p
                for p in self._operation.params)
            if len(self.free_parameters) == 0:
                # We can convert to a normal operation
                op_params = tuple(sympy_to_native(p) for p in new_op.params)
                new_op = new_op.set_params(op_params).substitute()
            return new_op.at(*self.qubits)
        else:
            return self._operation.at(*self.qubits)

    @property
    def qubits(self):
        return self._qubits

    @property
    def gates(self):
        return self,

    def operations(self):
        yield self

    @property
    def params(self):
        return self._params

    @property
    def free_parameters(self):
        return set().union(*(expr.free_symbols
                             for expr in self._params.values()))

    def set(self, **kwargs):
        kwargs = {key: sympify(val, locals=self._sympify_locals)
                  for key, val in kwargs.items()}
        self._params = {k: v.subs(kwargs) for k, v in self._params.items()}

    def __call__(self, **kwargs):
        new_gate = copy(self)
        new_gate.set(**kwargs)
        return new_gate

    @staticmethod
    def _operation_params(op):
        out = set()
        for op, _ in op.units():
            if isinstance(op, ParametrizedOperation):
                out.update(filter(lambda p: isinstance(p, str), op.params))
        return out


class Circuit(GateSetMixin):
    def __init__(self, qubits, gates):
        self._gates = list(gates)
        self._qubits = list(qubits)
        self._params_cache = None
        self._operation = None
        self._time_start = min((g.time_start for g in self._gates))
        self._time_end = max((g.time_end for g in self._gates))

    def __copy__(self):
        other = Circuit(self.qubits, (copy(gate) for gate in self._gates))
        other._params_cache = self._params_cache
        return other

    @property
    def qubits(self):
        return self._qubits

    @property
    def gates(self):
        return self._gates

    def operations(self):
        operations = flatten((gate.operations() for gate in self._gates))
        yield from sorted(operations, key=lambda op: op.time_start)

    @property
    def free_parameters(self):
        if self._params_cache is None:
            self._params_cache = set(chain(*(g.free_parameters
                                             for g in self._gates)))
        return self._params_cache

    def set(self, **kwargs):
        for gate in self._gates:
            gate.set(**kwargs)
        self._params_cache = None

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


class Box(CircuitUnitMixin, Circuit):
    """Several gates, united in one for logical purposes."""
    def __init__(self, qubits, gates, plot_metadata=None, repr_=None):
        super().__init__(qubits, gates)
        if plot_metadata:
            self.plot_metadata = plot_metadata
        if repr_:
            self._repr = repr_

    def __copy__(self):
        other = Box(self._qubits, (copy(gate) for gate in self._gates),
                    plot_metadata=deepcopy(self.plot_metadata),
                    repr_=self._repr)
        other._params_cache = self._params_cache
        return other

    @property
    def gates(self):
        return self,

    @property
    def params(self):
        out = {}
        for p in self.operations():
            out.update(p.params)
        return out

class FinalizedCircuit:
    """
    Parameters
    ----------
    operation : Operation
        An operation, that forms a final circuit
    qubits : list of str
        List of qubits in the operation
    """
    def __init__(self, qubits, gates, *, bases_in=None, sv_cutoff=1e-5):
        self.qubits = list(qubits)
        self.gates = list(gates)
        operations = [gate.operation_sympified() for gate in self.gates]
        # NB: operation must have sympy expressions in it
        self._params = set()
        units = []
        for op, ix in operations:
            self._params.update(self._op_params(op))
            new_ix = [self.qubits.index(qubit) for qubit in ix]
            units.append(op.at(*new_ix))
        if len(operations) > 0:
            self.operation = Operation.from_sequence(units) \
                                      .compile(bases_in=bases_in, sv_cutoff=sv_cutoff)
        else:
            self.operation = None

    @staticmethod
    def _op_params(op):
        if not isinstance(op, ParametrizedOperation):
            return set()
        # All parameters must be sympy expressions at this point
        return set(map(str, chain(*(p.free_symbols for p in op.params))))

    @classmethod
    def _deparametrize(cls, op, params=None):
        """
        Convert parametrized operation with sympy expressions, that can be
        converted into number into an operation.

        Parameters
        ----------
        op: ParametrizedOperation
        params: dict
            Extra parameters to set before instantiating the operation

        Returns
        -------
        Operation
        """
        if not isinstance(op, ParametrizedOperation):
            return op
        op_params = op.params
        if params:
            op_params = (p.subs(params) for p in op_params)
        op_params = tuple(sympy_to_native(p) for p in op_params)
        return op.set_params(op_params).substitute()

    def __call__(self, **params):
        if len(self._params) > 0:
            unset_params = self._params - params.keys()
            if len(unset_params) != 0:
                raise KeyError(*unset_params)
            units = [self._deparametrize(op, params).at(*ix)
                     for op, ix in self.operation.units()]
            out = FinalizedCircuit(self.qubits, [])
            out.operation = Operation.from_sequence(units).compile()
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
