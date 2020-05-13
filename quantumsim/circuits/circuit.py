import inspect
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from copy import copy, deepcopy
from itertools import chain
from math import log

import numpy as np
from pytools import flatten
from scipy.linalg import expm
from sympy import symbols, sympify

from .. import bases
from ..algebra import kraus_to_ptm
from ..algebra.algebra import plm_hamiltonian_part, plm_lindbladian_part
from ..operations.operation import Operation, ParametrizedOperation, \
    OperationNotDefinedError
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
    # noinspection PyTypeChecker
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


class PTMNotDefinedError(RuntimeError):
    pass


class GateSetMixin(ABC):
    """Abstract class, that defines an interface for all gates and
    circuits manipulation."""

    @abstractmethod
    def __copy__(self):
        pass

    @property
    @abstractmethod
    def dim_hilbert(self):
        """Hilbert dimensionality of qubits the operation acts onto."""
        pass

    @property
    @abstractmethod
    def qubits(self):
        """Qubit names, associated with this circuit."""
        pass

    @property
    @abstractmethod
    def time_start(self):
        """Starting time of a gate or a circuit."""
        pass

    @time_start.setter
    @abstractmethod
    def time_start(self, time):
        pass

    @property
    @abstractmethod
    def time_end(self):
        """Ending time of a gate or a circuit."""
        pass

    @time_end.setter
    @abstractmethod
    def time_end(self, time):
        pass

    @property
    @abstractmethod
    def duration(self):
        """Duration of the circuit."""
        pass

    @property
    @abstractmethod
    def gates(self):
        """Gates (logical units) of this circuit."""
        pass

    def operations(self):
        """Generator of operations (Mathematical units) of this circuit.

        Yields
        ------
        Gate
            Operations in chronological order
        """
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

    def __matmul__(self, state):
        """

        Parameters
        ----------
        state : quantumsim.State
        """
        # To ensure that all indices are present, so that exception is raised before
        # the computation, if there is a mistake.
        _ = self._qubit_indices_in_state(state)
        for op in self.operations():
            op @ state

    def _qubit_indices_in_state(self, state):
        if len(self.free_parameters) != 0:
            raise KeyError(*self.free_parameters)
        try:
            return [state.qubits.index(q) for q in self.qubits]
        except ValueError as ex:
            raise ValueError(
                'Qubit {} is not present in the state'
                .format(ex.args[0].split()[0]))

    @abstractmethod
    def ptm(self, bases_in, bases_out=None):
        """Return a Pauli transfer matrix, correspondent to this circuit.

        Returns
        -------
        array

        Raises
        ------
        PTMNotDefinedError
            If PTM can not yet be computed, becuse the Circuit has free parameters.
        """
        pass

    def finalize(self, bases_in=None):
        """
        Returns an optimized version of the circuit, that can be used to
        apply to the state.

        It will compile together all gates, that do not have params.
        `preprocessors` can be used to adjust the operation before compiling,
        for example, to instantiate custom placeholders, defined by the model.

        Parameters
        ----------
        bases_in : tuple of quantumsim.bases.PauliBasis

        Returns
        -------
        FinalizedCircuit
            Finalized version of this circuit
        """
        # noinspection PyTypeChecker
        return FinalizedCircuit(self.qubits, self.operations(), bases_in=bases_in)

    def _qubit_time_start(self, qubit):
        for gate in self.gates:
            if qubit in gate.qubits:
                return gate.time_start

    def _qubit_time_end(self, qubit):
        for gate in reversed(self.gates):
            if qubit in gate.qubits:
                return gate.time_end

    def _validate_bases(self, **kwargs):
        for name, bases in kwargs.items():
            if not hasattr(bases, '__iter__'):
                raise ValueError(
                    "`{n}` must be list-like, got {t}."
                    .format(n=name, t=type(bases)))
            if len(self.qubits) != len(bases):
                raise ValueError("Number of basis elements in `{}` ({}) does "
                                 "not match number of qubits in the "
                                 "operation ({})."
                                 .format(name, len(bases), len(self.qubits)))
            for b in bases:
                if self.dim_hilbert != b.dim_hilbert:
                    raise ValueError(
                        "Expected bases with Hilbert dimensionality {}, "
                        "but {} has elements with Hilbert dimensionality {}."
                        .format(self.dim_hilbert, name, b.dim_hilbert))


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

    def __init__(self, qubits, dim_hilbert, operation_func, duration=0., time_start=0.,
                 plot_metadata=None, repr_=None, bases_in=None, bases_out=None):
        """Gate

        Parameters
        ----------
        qubits : list of hashable or hashable
            Tags of the involved qubits
        dim_hilbert : int
            Hilbert dimensionality of the correspondent operations
        operation_func : callable
            A function, that returns a tuple of `(ptm, bases_in, bases_out)`,
            where PTM is a Pauli transfer matrix in basis `bases_in`, `bases_out`.
        duration: float
            Duration of the operation.
        time_start: float
            Starting time of the operation.
        plot_metadata : None or dict
            Metadata, that describes how to represent a gate on a plot.
            TODO: link documentation, when plotting is ready.
        repr_ : None or str
            Pretty-printable representation of the gate, used in `Gate.__repr__`
            and `Gate.__str__`. Can contain Python formatting syntax, then parameters
            are picked from the Gate parameters when displayed.
            If `None`, defaults to `"gate"`.
        bases_in: tuple of quantumsim.bases.PauliBasis or None
            Input bases for the operation. Used to force reduce the basis set to a
            subset of full basis, for example for the purpose of constructing
            a dephasing operation. If None, assumed to be default full basis.
        bases_out: tuple of quantumsim.bases.PauliBasis or None
            Input bases for the operation. Used to force reduce the basis set to a
            subset of full basis, for example for the purpose of constructing
            a dephasing operation. If None, assumed to be the same as `bases_in`.
        """
        super(Gate, self).__init__()
        self._dim_hilbert = dim_hilbert
        if hasattr(qubits, '__iter__') and not isinstance(qubits, str):
            self._qubits = tuple(qubits)
        else:
            self._qubits = (qubits,)

        argspec = inspect.getfullargspec(operation_func)
        if argspec.varargs is not None:
            raise ValueError(
                "`operation_func` can't accept free arguments.")
        if argspec.varkw is not None:
            raise ValueError(
                "`operation_func` can't accept free keyword arguments.")
        self._operation_func = operation_func
        # OrderedDict is mostly to emphacise that order is important here,
        # since Python 3.7 dict is guaranteed to be ordered
        self._params = OrderedDict(((param, symbols(param)) for param in argspec.args))

        if plot_metadata:
            self.plot_metadata = plot_metadata
        if repr_:
            self._repr = repr_
        self._duration = duration
        self._time_start = time_start

        # Shim layer: we still construct an operation internally
        def op(*args):
            return Operation.from_ptm(*self._operation_func(*args))

        op.__signature__ = inspect.signature(self._operation_func)
        bases_in = bases_in or (bases.general(self._dim_hilbert),) * len(self.qubits)
        bases_out = bases_out or bases_in
        self._operation = ParametrizedOperation(op, bases_in, bases_out)

    def __copy__(self):
        other = Gate(
            self._qubits, self.dim_hilbert, self._operation_func,
            self._duration, self._time_start, self.plot_metadata, self._repr)
        other._params = copy(self._params)
        return other

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

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
    def bases_in(self):
        return self._operation.bases_in

    @property
    def bases_out(self):
        return self._operation.bases_out

    @property
    def duration(self):
        return self._duration

    @classmethod
    def from_ptm(cls, ptm, bases_in, bases_out=None, *, qubits=None,
                 duration=0., time_start=0., plot_metadata=None, repr_=None):
        """ Construct an operation from a Pauli transfer matrix provided in
        a certain Pauli basis.

        Parameters
        ----------
        ptm: array-like
            Pauli transfer matrix in a form of Numpy array.
        bases_in: tuple of quantumsim.bases.PauliBasis
            Input bases of qubits.
        bases_out: tuple of quantumsim.bases.PauliBasis
            Output bases of qubits. If None, assumed to be the same as input
            bases.
        qubits: list of hashable, hashable or None
            List of qubit tags for the operation. Tags must be able to serve as `dict` keys.
            If `None`, integer tags are picked starting from 0.
        duration: float
            Duration of the operation.
        time_start: float
            Starting time of the operation.
        plot_metadata : None or dict
            Metadata, that describes how to represent a gate on a plot.
            TODO: link documentation, when plotting is ready.
        repr_ : None or str
            Pretty-printable representation of the gate, used in `Gate.__repr__`
            and `Gate.__str__`. Can contain Python formatting syntax, then parameters
            are picked from the Gate parameters when displayed.
            If `None`, defaults to `"gate"`.

        Returns
        -------
        Gate
            Resulting operation
        """
        if bases_out is None:
            bases_out = bases_in
        num_qubits = len(bases_in)
        if qubits is None:
            qubits = list(range(num_qubits))
        return Gate(qubits, bases_in[0].dim_hilbert, lambda: (ptm, bases_in, bases_out),
                    duration, time_start, plot_metadata, repr_, bases_in, bases_out)

    @classmethod
    def from_kraus(cls, kraus, bases_in=2, bases_out=None, *, qubits=None,
                   duration=0, time_start=0., plot_metadata=None, repr_=None):
        """Construct an operation from a set of Kraus matrices.

        TODO: elaborate on Kraus matrices format.

        Parameters
        ----------
        kraus: array-like or function
            Pauli transfer matrix in a form of Numpy array. If a function is
            provided, it must return a Kraus and its parameters are treated as gate's
            named parameters.
        bases_in : tuple of PauliBasis or int
            Input bases for generated PTMs. If `int` is provided, it is treated as a
            Hilbert space dimensionality, and default basis is picked accordingly to it.
        bases_out : tuple of PauliBasis or None
            Output bases for generated PTMs. If None, defaults to `bases_in`.
        qubits: list of hashable, hashable or None
            List of qubit tags for the operation. Tags must be able to serve as `dict` keys.
            If `None`, integer tags are picked starting from 0.
        duration: float
            Duration of the operation.
        time_start: float
            Starting time of the operation.
        plot_metadata : None or dict
            Metadata, that describes how to represent a gate on a plot.
            TODO: link documentation, when plotting is ready.
        repr_ : None or str
            Pretty-printable representation of the gate, used in `Gate.__repr__`
            and `Gate.__str__`. Can contain Python formatting syntax, then parameters
            are picked from the Gate parameters when displayed.
            If `None`, defaults to `"gate"`.

        Returns
        -------
        Gate
            Resulting operation
        """
        if not isinstance(kraus, np.ndarray):
            kraus = np.array(kraus)
        if len(kraus.shape) == 2:
            kraus = kraus.reshape((1, *kraus.shape))
        elif len(kraus.shape) != 3:
            raise ValueError(
                '`kraus` should be a 2D or 3D array, got shape {}'
                .format(kraus.shape))
        kraus_size = kraus.shape[1]
        if isinstance(bases_in, int):
            dim_hilbert = bases_in
            bases_in = cls._default_bases(kraus_size, dim_hilbert)
            bases_out = bases_out or bases_in
        else:
            dim_hilbert = bases_in[0].dim_hilbert
            num_qubits = len(bases_in)
            if (kraus_size != dim_hilbert ** num_qubits or
                    kraus_size != kraus.shape[2]):
                raise ValueError('Shape of the Kraus operator for bases provided must '
                                 'be {0}x{0}, got {1}x{2} instead'
                                 .format(dim_hilbert ** num_qubits, kraus.shape[1],
                                         kraus.shape[2]))
        bases_out = bases_out or bases_in
        return cls.from_ptm(kraus_to_ptm(kraus, bases_in, bases_out),
                            bases_in, bases_out, qubits=qubits,
                            duration=duration, time_start=time_start,
                            plot_metadata=plot_metadata,
                            repr_=repr_)

    @classmethod
    def from_lindblad_form(cls, time, bases_in, bases_out=None, *,
                           hamiltonian=None, lindblad_ops=None, qubits=None,
                           duration=None, time_start=0., plot_metadata=None,
                           repr_=None):
        """Construct and operation from a list of Lindblad operators.

        TODO: elaborate on Lindblad operators format

        Parameters
        ----------
        time : float
            Duration of an evolution, driven by Lindblad equation,
            in arbitrary units.
        bases_in : tuple of PauliBasis or int
            Input bases for generated PTMs. If `int` is provided, it is treated as a
            Hilbert space dimensionality, and default basis is picked accordingly to it.
        bases_out : tuple of PauliBasis or None
            Output bases for generated PTMs. If None, defaults to `bases_in`.
        hamiltonian: array or None
            Hamiltonian for a Lindblad equation. In units :math:`\\hbar = 1`.
            If `None`, assumed to be zero.
        lindblad_ops: array or list of arrays
            Lindblad jump operators. In units :math:`\\hbar = 1`.
            If `None`, assumed to be zero.

        Returns
        -------
        Gate
        """
        summands = []
        if hamiltonian is not None:
            if isinstance(bases_in, int):
                bases_in = cls._default_bases(len(hamiltonian), bases_in)
            summands.append(plm_hamiltonian_part(hamiltonian, bases_in))
        if lindblad_ops is not None:
            if isinstance(bases_in, int):
                bases_in = cls._default_bases(len(hamiltonian), bases_in)
            if isinstance(lindblad_ops, np.ndarray) and \
                    len(lindblad_ops.shape) == 2:
                lindblad_ops = (lindblad_ops,)
            if not isinstance(lindblad_ops, np.ndarray):
                lindblad_ops = np.array(lindblad_ops)
            summands.append(plm_lindbladian_part(lindblad_ops, bases_in))
        if len(summands) == 0:
            raise ValueError("Either `hamiltonian` or `lindblad_ops` must be "
                             "provided.")
        plm = np.sum(summands, axis=0) * time
        dim = np.prod(plm.shape[:len(plm.shape) // 2])
        ptm = expm(plm.reshape((dim, dim))).reshape(plm.shape)
        if not np.allclose(ptm.imag, 0):
            raise ValueError('Resulting PTM is not real-valued, check the '
                             'sanity of `hamiltonian` and `lindblad_ops`.')
        out = cls.from_ptm(ptm.real, bases_in, bases_in, qubits=qubits,
                           duration=duration, time_start=time_start,
                           plot_metadata=plot_metadata, repr_=repr_)
        if bases_out is not None:
            return out.set_bases(bases_out=bases_out)
        else:
            return out

    # noinspection PyUnresolvedReferences
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
        self._params = {k: v.subs(kwargs, simultaneous=True)
                        for k, v in self._params.items()}

    def __call__(self, **kwargs):
        new_gate = copy(self)
        new_gate.set(**kwargs)
        return new_gate

    @classmethod
    def _default_bases(cls, kraus_size, dim_hilbert):
        num_qubits = int(log(kraus_size, dim_hilbert))
        if not dim_hilbert**num_qubits != kraus_size:
            raise ValueError("Computed number of qubits in the operation is not "
                             "integer.")
        return (bases.general(dim_hilbert),) * num_qubits

    def set_bases(self, bases_in=None, bases_out=None):
        """Return an version of this circuit with the input and output
        bases updated.

        This is useful, if a user wants to truncate a basis set. For example,
        depolarizing operation is obtained from the identity by setting its basis to
        a classical subbasis. Also, this function is heavily used internally during
        the circuit compilation.

        Parameters
        ----------
        bases_in : tuple of PauliBasis or None
            Input bases of the qubits. If `None` provided, full :math:`01xy`
            basis is assumed.
        bases_out : tuple of PauliBasis or None
            Output bases of the qubits. If `None` provided, full :math:`01xy`
            basis is assumed.

        Returns
        -------
        quantumsim.operations.Operation
            Equivalent operation in the new basis.
        """
        if bases_in is not None:
            self._validate_bases(bases_in=bases_in)
        if bases_out is not None:
            self._validate_bases(bases_out=bases_out)

        other = copy(self)
        other._operation = self._operation.set_bases(bases_in, bases_out)
        return other

    def ptm(self, bases_in, bases_out=None):
        try:
            return self.operation_sympified().operation.ptm(bases_in, bases_out)
        except OperationNotDefinedError as ex:
            raise PTMNotDefinedError from ex

    def __matmul__(self, state):
        """

        Parameters
        ----------
        state : quantumsim.State

        Raises
        ------
        PTMNotDefinedError
            If the gate has free parameters
        """
        # To ensure that all indices are present, so that exception is raised before
        # the computation, if there is a mistake.
        indices = self._qubit_indices_in_state(state)
        try:
            self.operation_sympified().operation(state.pauli_vector, *indices)
        except OperationNotDefinedError as ex:
            raise PTMNotDefinedError from ex


class Circuit(GateSetMixin):
    @property
    def dim_hilbert(self):
        # FIXME: there should be some validation and caching
        return self._gates[0].dim_hilbert

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

    def ptm(self, bases_in, bases_out=None):
        def qtoi(op_ix):
            op, ix = op_ix
            return op.at(*(self._qubits.index(q) for q in ix))

        try:
            return Operation.from_sequence(
                list(qtoi(op.operation_sympified()) for op in self.operations()))\
                .ptm(bases_in, bases_out)
        except OperationNotDefinedError as ex:
            raise PTMNotDefinedError from ex


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
    qubits : list of str
        List of qubits in the operation
    gates : list of Gate
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
            op_params = (p.subs(params, simultaneous=True) for p in op_params)
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
