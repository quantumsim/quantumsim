import abc
import inspect
import re

import numpy as np
import scipy.linalg.matfuncs
from collections import namedtuple
from itertools import chain

from copy import copy

from ..algebra.algebra import (kraus_to_ptm, ptm_convert_basis,
                               plm_lindbladian_part, plm_hamiltonian_part)
from ..bases import PauliBasis


class OperationNotDefinedError(RuntimeError):
    pass


class Operation(metaclass=abc.ABCMeta):
    """A metaclass for all quantum operations.

    Every operation has to implement call method, that takes a
    :class:`quantumsim.pauli_vectors.PauliVectorBase` object and modifies it
    inline.
    """

    @property
    def _default_compiler_cls(self):
        from .compiler import ChainCompiler
        return ChainCompiler

    @property
    @abc.abstractmethod
    def dim_hilbert(self):
        """Hilbert dimensionality of qubits the operation acts onto."""
        pass

    @property
    @abc.abstractmethod
    def num_qubits(self):
        """Hilbert dimensionality of qubits the operation acts onto."""
        pass

    @abc.abstractmethod
    def __call__(self, pauli_vector, *qubits):
        """Applies the operation inline (modifying the state) to the Pauli
        vector. Number of qubit indices should be aligned with a
        dimensionality of the operation.

        Parameters
        ----------
        pauli_vector : quantumsim.pauli_vectors.PauliVectorBase
            A Pauli vector, representing the state of qubits
        q0, ..., qN : int
            Indices of a qubit in a state
        """
        pass

    @abc.abstractmethod
    def set_bases(self, bases_in=None, bases_out=None):
        """Return an version of this operation with the input and output
        bases set to provided in arguments or unchanged.

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
        if bases_in is None and bases_out is None:
            raise ValueError('Either bases_in or bases_out must be specified.')
        if bases_in is not None:
            self._validate_bases(bases_in=bases_in)
        if bases_out is not None:
            self._validate_bases(bases_out=bases_out)

    @abc.abstractmethod
    def ptm(self, bases_in, bases_out=None):
        """Return a full Pauli transfer matrix of the operation.

        Parameters
        ----------
        bases_in : tuple of PauliBasis
            Input basis of the PTM
        bases_out : tuple of PauliBasis or None
            Output bases of the PTM. If None, deraults to bases_in

        Returns
        -------
        ptm : ndarray
            Pauli transfer matrix in bases specified.
        """
        self._validate_bases(bases_in=bases_in)
        if bases_out is not None:
            self._validate_bases(bases_out=bases_out)

    @staticmethod
    def from_ptm(ptm, bases_in, bases_out=None):
        """Construct completely positive map, based on Pauli transfer matrix.

        TODO: elaborate on PTM format.

        Parameters
        ----------
        ptm: ndarray
            Pauli transfer matrix in a form of Numpy array
        bases_in: tuple of quantumsim.bases.PauliBasis
            Input bases of qubits.
        bases_out: tuple of quantumsim.bases.PauliBasis
            Output bases of qubits. If None, assumed to be the same as input
            bases.

        Returns
        -------
        Transformation
            Resulting operation
        """
        if bases_out is None:
            bases_out = bases_in
        return _PTMOperation(ptm, bases_in=bases_in, bases_out=bases_out)

    @staticmethod
    def from_kraus(kraus, bases_in, bases_out=None):
        """Construct an operation from a set of Kraus matrices.

        Either bases, or `dim_hilbert` must be specified.

        TODO: elaborate on Kraus matrices format.

        Parameters
        ----------
        kraus: ndarray
            Pauli transfer matrix in a form of Numpy array
        bases_in : tuple of PauliBasis
            Input bases for generated PTMs. If None, default is picked.
        bases_out : tuple of PauliBasis or None
            Output bases for generated PTMs. If None, defaults to `bases_in`.

        Returns
        -------
        Transformation
            Resulting operation
        """
        bases_out = bases_out or bases_in
        if not isinstance(kraus, np.ndarray):
            kraus = np.array(kraus)
        if len(kraus.shape) == 2:
            kraus = kraus.reshape((1, *kraus.shape))
        elif len(kraus.shape) != 3:
            raise ValueError(
                '`kraus` should be a 2D or 3D array, got shape {}'
                .format(kraus.shape))

        dim_hilbert = bases_in[0].dim_hilbert
        num_qubits = len(bases_in)
        kraus_size = kraus.shape[1]
        if (kraus_size != dim_hilbert ** num_qubits or
                kraus_size != kraus.shape[2]):
            raise ValueError(
                'Shape of the Kraus operator for bases provided must be '
                '{0}x{0}, got {1}x{2} instead'
                .format(dim_hilbert ** num_qubits,
                        kraus.shape[1], kraus.shape[2]))

        return Operation.from_ptm(kraus_to_ptm(kraus, bases_in, bases_out),
                                  bases_in, bases_out)

    @staticmethod
    def from_lindblad_form(time, basis, *, hamiltonian=None, lindblad_ops=None):
        """Construct and operation from a list of Lindblad operators.

        TODO: elaborate on Lindblad operators format

        Parameters
        ----------
        time : float
            Duration of an evolution, driven by Lindblad equation,
            in arbitrary units.
        basis: quantumsim.bases.PauliBasis
            A basis for the resulting operation.
        hamiltonian: array or None
            Hamiltonian for a Lindblad equation. In units :math:`\\hbar = 1`.
            If `None`, assumed to be zero.
        lindblad_ops: array or list of arrays
            Lindblad jump operators. In units :math:`\\hbar = 1`.
            If `None`, assumed to be zero.

        Returns
        -------
        quantumsim.operations.operation._PTMOperation
        """
        summands = []
        if hamiltonian is not None:
            summands.append(plm_hamiltonian_part(hamiltonian, basis, basis))
        if lindblad_ops is not None:
            if isinstance(lindblad_ops, np.ndarray) and \
                    len(lindblad_ops.shape) == 2:
                lindblad_ops = (lindblad_ops,)
            for op in lindblad_ops:
                summands.append(plm_lindbladian_part(op, basis, basis))
        if len(summands) == 0:
            raise ValueError("Either `hamiltonian` or `lindblad_ops` must be "
                             "provided.")
        ptm = scipy.linalg.matfuncs.expm(np.sum(summands, axis=0) * time)
        if not np.allclose(ptm.imag, 0):
            raise ValueError('Resulting PTM is not real-valued, check the '
                             'sanity of `hamiltonian` and `lindblad_ops`.')
        return _PTMOperation(ptm.real, (basis,), (basis,))

    @staticmethod
    def from_sequence(*operations):
        """
        Constructs an operation from a sequence of operations, that are
        applied in the given order.

        Parameters
        ----------
        op0, ..., opN: Operation, _IndexedOperation or list
            Operations to concatenate, or a single list of operations. If not
            all operations match in qubit dimensionality or in order of qubits
            to be applied to, they must be indexed
            (see :func:`Operation.at` method).

        Returns
        -------
        quantumsim.operations.operation._Chain
            Resulting operation
        """
        if not (isinstance(operations[0], _IndexedOperation) or
                isinstance(operations[0], Operation)):
            if hasattr(operations[0], '__iter__'):
                operations = operations[0]
            else:
                raise ValueError(
                    "Wrong type of operation number 0: {}"
                    .format(type(operations[0])))

        op0 = operations[0]
        if isinstance(op0, Operation):
            for i, op in enumerate(operations[1:], 1):
                if isinstance(op, _IndexedOperation):
                    raise ValueError(
                        "Provide index for operation number 0 (see "
                        "`Operation.at()` method).")
                if not isinstance(op, Operation):
                    raise ValueError(
                        "Wrong type of operation number {}: {}"
                        .format(i, type(op)))
                if op0.dim_hilbert != op.dim_hilbert:
                    raise ValueError(
                        "Hilbert dimensionality of operation number 0 ({}) "
                        "does not match with Hilbert dimensionality of "
                        "operation number {} ({})"
                        .format(op0.dim_hilbert, i, op.dim_hilbert))
                if op0.num_qubits != op.num_qubits:
                    raise ValueError(
                        "Number of qubits in operation 0 ({}) does not match "
                        "with a number of qubits in operation {} ({}). "
                        "Provide indices explicitly (see `Operation.at()` "
                        "method).".format(op0.num_qubits, i, op.num_qubits))
        else:
            # op0 is certainly _IndexedOperation, we checked
            for i, op in enumerate(operations[1:], 1):
                if isinstance(op, Operation):
                    raise ValueError(
                        "Provide index for operation number {} (see "
                        "`Operation.at()` method).".format(i))
                if not isinstance(op, _IndexedOperation):
                    raise ValueError(
                        "Wrong type of operation number {}: {}"
                        .format(i, type(op)))
                if op0.operation.dim_hilbert != op.operation.dim_hilbert:
                    raise ValueError(
                        "Hilbert dimensionality of operation number 0 ({}) "
                        "does not match with Hilbert dimensionality of "
                        "operation number {} ({})"
                        .format(op0.operation.dim_hilbert, i,
                                op.operation.dim_hilbert))

        if isinstance(operations[0], Operation):
            indices = tuple(range(operations[0].num_qubits))
            operations = [op.at(*indices) for op in operations]

        return _Chain(operations)

    def compile(self, bases_in=None, bases_out=None, *, compiler_cls=None):
        """Returns equivalent circuit, optimized for given input and/or
        output bases.

        `bases_in` should match the basis of a Pauli vector, to which
        operation is applied. If the state bases are not subbases of an
        operation, or operation itself is a compiled operation with a reduced
        basis, and `bases_in` or `bases_out` are not the subbases of those,
        for which operation is compiled for, applying an operation will
        silently produce wrong result. We advice to use this function only
        on operations, that were not yet compiled.

        Parameters
        ----------
        bases_in: None or list of quantumsim.bases.PauliBasis
            Input bases.
        bases_out: None or list of quantumsim.bases.PauliBasis
            Output bases.
        compiler_cls: none or class
            Class of a compiler. If None, Quantumsim decides.

        Returns
        -------
        quantumsim.operations.operation._Chain or
        quantumsim.operations.operation._PTMOperation
        """
        if isinstance(self, _Chain):
            op = self
        else:
            op = Operation.from_sequence(self)
        compiler_cls = compiler_cls or self._default_compiler_cls
        compiler = compiler_cls(op, optimize=True)
        return compiler.compile(bases_in, bases_out)

    def at(self, *indices):
        """Returns a container with the operation, that provides also dumb
        indices of qubits it acts on. Used during processes' concatenation
        to match qubits they act onto.

        Parameters
        ----------
        i0, ..., iN: int
            Dumb indices of qubits operation acts onto.

        Returns
        -------
        _IndexedOperation
            Intermediate representation of an operation.
        """
        if not self.num_qubits == len(indices):
            raise ValueError('Number of indices is not equal to the number of '
                             'qubits in the operation.')
        return _IndexedOperation(self, indices)

    def _validate_bases(self, **kwargs):
        for name, bases in kwargs.items():
            if not hasattr(bases, '__iter__'):
                raise ValueError(
                    "`{n}` must be list-like, got {t}."
                    .format(n=name, t=type(bases)))
            if self.num_qubits != len(bases):
                raise ValueError("Number of basis elements in `{}` ({}) does "
                                 "not match number of qubits in the "
                                 "operation ({})."
                                 .format(name, len(bases), self.num_qubits))
            for b in bases:
                if self.dim_hilbert != b.dim_hilbert:
                    raise ValueError(
                        "Expected bases with Hilbert dimensionality {}, "
                        "but {} has elements with Hilbert dimensionality {}."
                        .format(self.dim_hilbert, name, b.dim_hilbert))


_IndexedOperation = namedtuple('_IndexedOperation', ['operation', 'indices'])


class Placeholder(Operation):
    """
    Parameters
    ----------
    bases_in: tuple of quantumsim.bases.PauliBasis
        Input bases of qubits.
    bases_out: tuple of quantumsim.bases.PauliBasis or None
        Output bases of qubits

    Returns
    -------
    A placeholder for an operation
    """
    def __init__(self, bases_in, bases_out=None):
        self._num_qubits = len(bases_in)
        self._dim_hilbert = bases_in[0].dim_hilbert
        self._validate_bases(bases_in=bases_in)
        if bases_out is None:
            bases_out = bases_in
        else:
            self._validate_bases(bases_out=bases_out)
        self._bases_in = bases_in
        self._bases_out = bases_out

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def bases_in(self):
        return self._bases_in

    @property
    def bases_out(self):
        return self._bases_out

    def __call__(self, pauli_vector, *qubits):
        raise OperationNotDefinedError(
            'Operation placeholder can not be called')

    def set_bases(self, bases_in=None, bases_out=None):
        super().set_bases(bases_in, bases_out)
        b_in = bases_in or self.bases_in
        b_out = bases_out or self.bases_out
        if b_in == self.bases_in and b_out == self.bases_out:
            return self
        new_op = copy(self)
        new_op._bases_in = b_in
        new_op._bases_out = b_out
        return new_op

    def ptm(self, bases_in, bases_out=None):
        raise OperationNotDefinedError(
            'Operation placeholder does not have a PTM')


class _PTMOperation(Operation):
    """Generic transformation of a state.

    Any transformation, that should be a completely positive map, can be
    represented in a form of set of Kraus operators [1]_ or as a Pauli transfer
    matrix [2]_. Dependent on your preferred representation, you may construct
    the transformation with :func:`Operation.from_kraus` or
    :func:`Operation.from_ptm`. Constructor of this class is not
    supposed to be called in user code.

    Parameters
    ----------
    ptm : ndarray
        Pauli transfer matrix of an operation.
    bases_in : tuple of PauliBasis
        Input bases of the PTM
    bases_out : tuple of PauliBasis
        Output bases of the PTM

    References
    ----------
    .. [1] M. A. Nielsen, I. L. Chuang, "Quantum Computation and Quantum
       Information" (Cambridge University Press, 2000).
    .. [2] D. Greenbaum, "Introduction to Quantum Gate Set Tomography",
       arXiv:1509.02921 (2000).
    """

    def __init__(self, ptm, bases_in, bases_out):
        self._ptm = ptm
        self.bases_in = bases_in
        self.bases_out = bases_out
        self._dim_hilbert = bases_in[0].dim_hilbert
        self._num_qubits = len(self.bases_in)
        self._validate_bases(bases_out=self.bases_out)
        shape = tuple(b.dim_pauli for b in
                      chain(self.bases_out, self.bases_in))
        if not ptm.shape == shape:
            raise ValueError(
                'Shape of `ptm` is not compatible with the `bases` '
                'dimensionality: \n'
                '- expected shape from provided `bases`: {}\n'
                '- `ptm` shape: {}'.format(shape, ptm.shape))

    @property
    def dim_hilbert(self):
        """Returns Hilbert dimensionality of an operation."""
        return self._dim_hilbert

    @property
    def shape(self):
        """Shape of a PTM, that represents the operation, qubit-wise.

        For example, for a single-qubit gate, acting in a full basis,
        shape should be :math:`\\left(d^2, d^2\\right)`, where d is a Hilbert
        dimensionality of a qubit subspace. First item corresponds to the
        output dimensionality, second -- to the input one. For a two-qubit
        gate it will be :math:`\\left(d^2, d^2, d^2, d^2\\right)`.
        If PTM acts on a reduced basis or reduces a basis (for example,
        it is a projection), elements can be less than :math:`d^2`.
        """
        return self._ptm.shape

    @property
    def num_qubits(self):
        """Returns number of qubits an operation involves."""
        return self._num_qubits

    def set_bases(self, bases_in=None, bases_out=None):
        super().set_bases(bases_in, bases_out)
        b_in = bases_in or self.bases_in
        b_out = bases_out or self.bases_out
        if b_in == self.bases_in and b_out == self.bases_out:
            new_op = self
        else:
            new_ptm = ptm_convert_basis(self._ptm,
                                        self.bases_in, self.bases_out,
                                        b_in, b_out)
            new_op = _PTMOperation(new_ptm, b_in, b_out)
        return new_op

    def ptm(self, bases_in, bases_out=None):
        bases_out = bases_out or bases_in
        if bases_in == self.bases_in and bases_out == self.bases_out:
            return self._ptm
        else:
            return self.set_bases(bases_in, bases_out)._ptm

    def __call__(self, pauli_vector, *qubit_indices):
        """

        Parameters
        ----------
        pauli_vector : quantumsim.pauli_vectors.PauliVectorBase
        q0, ..., qN : indices of qubits to act on
        """
        if len(qubit_indices) != self.num_qubits:
            raise ValueError('This is a {}-qubit operation, but number of '
                             'qubits provided is {}'
                             .format(self.num_qubits, len(qubit_indices)))
        op = self
        for q, b in zip(qubit_indices, self.bases_in):
            if pauli_vector.bases[q] != b:
                op = self.set_bases(
                    bases_in=tuple([pauli_vector.bases[q] for q in qubit_indices]))
                break

        pauli_vector.apply_ptm(op._ptm, *qubit_indices)
        for q, b in zip(qubit_indices, op.bases_out):
            pauli_vector.bases[q] = b


class _Chain(Operation):
    """
    A chain of operations, that are applied sequentially.
    """

    def __init__(self, operations):
        self._dim_hilbert = operations[0].operation.dim_hilbert
        all_indices = np.unique(
            list(chain(*(op.indices for op in operations))))
        if all_indices[0] != 0 or all_indices[-1] != len(all_indices) - 1:
            raise ValueError('Indices of operations must form an ordered set '
                             'from 0 to N-1')
        self._num_qubits = len(all_indices)

        joined_ops = []
        for op_indices in operations:
            # Flatten the operations chain
            if isinstance(op_indices.operation, _Chain):
                for sub_op, sub_indices in op_indices.operation.operations:
                    _, indices = op_indices
                    new_indices = tuple((indices[i] for i in sub_indices))
                    joined_ops.append(_IndexedOperation(sub_op, new_indices))
            else:
                joined_ops.append(op_indices)

        self.operations = joined_ops
        for op in self.operations:
            if isinstance(op.operation, _Chain):
                raise RuntimeError('Chain must not contain chains; this is '
                                   'probably a bug.')

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def num_qubits(self):
        return self._num_qubits

    def __call__(self, pauli_vector, *qubit_indices):
        if len(qubit_indices) != self._num_qubits:
            raise ValueError('This is a {}-qubit operation, number of qubit '
                             'indices provided is {}'
                             .format(self._num_qubits, len(qubit_indices)))
        results = []
        for op, indices in self.operations:
            result = op(pauli_vector, *(qubit_indices[i] for i in indices))
            if result is not None:
                results.append(result)
        return results if len(results) > 0 else None

    def set_bases(self, bases_in=None, bases_out=None):
        super().set_bases(bases_in, bases_out)
        compiler = self._default_compiler_cls(self, optimize=False)
        return compiler.compile(bases_in, bases_out)

    def ptm(self, bases_in, bases_out=None):
        super().ptm(bases_in, bases_out)
        bases_out = bases_out or bases_in
        ptm_in_shape = tuple(b.dim_pauli for b in bases_in)
        start_ptm = Operation.from_ptm(
            np.identity(np.prod(ptm_in_shape), dtype=float)
            .reshape(ptm_in_shape*2), bases_in)
        return self._default_compiler_cls(
            Operation.from_sequence(start_ptm, self),
            optimize=True) \
            .compile(bases_in, bases_out) \
            .ptm(bases_in, bases_out)

    def substitute(self, **kwargs):
        operations = []
        operations = [_IndexedOperation(
            op.substitute(**kwargs) if isinstance(op, ParametrizedOperation)
            else op,
            ix) for op, ix in self.operations]
        return _Chain(operations)


class ParametrizedOperation(Placeholder):
    _valid_identifier_re = re.compile('[a-zA-Z_][a-zA-Z0-9_]*')

    def __init__(self, operation_func, bases_in, bases_out=None, **params):
        """A gate without notion of timing.

        Parameters
        ----------
        operation_func : function
            A function, that takes a certain number of arguments
            (gate parameters) and returns an operation.
        bases_in : tuple of quantumsim.PauliBasis
            Input bases, that will be taken by final, when it is calculated.
        bases_out : tuple of quantumsim.PauliBasis or None
            Output bases, that will be taken by final, when it is calculated.
            If None, the same as input basis is taken.
        **params:
            Parameters, that are set initially.
        """
        super().__init__(bases_in, bases_out)
        self._operation_func = operation_func
        argspec = inspect.getfullargspec(operation_func)
        if argspec.varargs is not None:
            raise ValueError(
                "`operation_func` can't accept free arguments.")
        if argspec.varkw is not None:
            raise ValueError(
                "`operation_func` can't accept free keyword arguments.")
        self._params_real = tuple(argspec.args)
        self._params = set(self._params_real)
        self._params_set = {k: v for k, v in params.items() if k in
                            self._params}
        self._params_subs = {}
        self._operation = None

    @property
    def params(self):
        return self._params

    def __copy__(self):
        copy_ = self.__class__(self._operation_func, self._bases_in,
                               self._bases_out)
        copy_._params = copy(self._params)
        copy_._params_set = copy(self._params_set)
        copy_._params_subs = copy(self._params_subs)
        return copy_

    def _substitute_param(self, name, value):
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

    def substitute(self, **kwargs):
        params_to_substitute = self._params.intersection(kwargs.keys())
        if len(params_to_substitute) == 0 and len(self._params) != 0:
            return self
        copy_ = copy(self)
        for name in params_to_substitute:
            copy_._substitute_param(name, kwargs[name])
        if len(copy_._params) == 0:
            for name, real_name in copy_._params_subs.items():
                copy_._params_set[real_name] = copy_._params_set.pop(name)
            args = tuple(copy_._params_set[name] for name in copy_._params_real)
            op = copy_._operation_func(*args)
            if not isinstance(op, Operation):
                raise RuntimeError(
                    'Invalid operation function was provided for the '
                    'parametrized operation during its creation: '
                    'it must raturn a quantumsim.Operation instance, '
                    'but it returns {} instead.'.format(type(op)))
            if op.num_qubits != copy_.num_qubits:
                raise RuntimeError(
                    'Invalid operation function was provided for the '
                    'parametrized operation during its creation: '
                    'its number of qubits ({}) does not match one of the '
                    'basis ({}).'.format(op.num_qubits,
                                         copy_.num_qubits))
            return op.set_bases(self._bases_in, self._bases_out)
        else:
            return copy_
