import abc
from ..state import State
from .common import kraus_to_ptm
from .common import _check_ptm_dims
from ..bases import general


class Operation(metaclass=abc.ABCMeta):
    """A metaclass for all operations.

    Every operation has to implement call method, that takes a
    :class:`qs2.state.State` object and modifies it inline. This method may
    return nothing or a result of a measurement, if the operation is a
    measurement.

    Operations are designed to form an algebra. For the sake of making
    interface sane, we do not provide explicit multiplication, instead we use
    :func:`qs2.operations.join` function to concatenate a set of operations.
    I.e., if we have `rotate90` operation, we may construct `rotate180`
    operation as follows:

    >>> import qs2, numpy
    ... rotY90 = qs2.operations.rotate_z(0.5*numpy.pi)
    ... rotY180 = qs2.operations.join(rotY90, rotY90)

    If the dimensionality of operations does not match, we may specify qubits it
    acts onto in a form of integer dumb indices:

    >>> cz = qs2.operations.cphase()
    ... cnot = qs2.operations.join(rotY90.at(0), cz.at(0, 1), rotY90.at(0))

    This is also useful, if you want to combine operations on different
    qubits into one:

    >>> hadamard = qs2.operations.hadamard()
    ... hadamard3q = qs2.operations.join(hadamard.at(0), hadamard.at(1),
    ...                                  hadamard.at(2))

    All the dumb indices involved in `join` function must form an ordered set
    `0, 1, ..., N`.

    Parameters
    ----------
    indices: None or tuple
        Indices of qubits it acts on. They are designed to be dumb and used
        for tracking the multiplication of several operations.
    """
    @abc.abstractmethod
    def __call__(self, state, *qubit_indices):
        """Applies the operation inline (modifying the state) to the state
        to certain qubits. Number of qubit indices should be aligned with a
        dimensionality of the operation.
        """
        pass

    def at(self, *indices):
        """Returns a container with the operation, that provides also dumb
        indices of qubits it acts on. Used during operations' concatenation
        to match qubits they act onto.

        Parameters
        ----------
        i0, ..., iN: int
            Dumb indices of qubits operation acts onto.

        Returns
        -------
        _DumbIndexedOperation
            Intermediate representation of an operation.
        """
        return _DumbIndexedOperation(self, indices)

    @property
    @abc.abstractmethod
    def n_qubits(self):
        pass


class _DumbIndexedOperation:
    """Internal representation of an operations during their multiplications.
    Contains an operation itself and dumb indices of qubits it acts on.
    """

    def __init__(self, operation, indices):
        self._operation = operation
        self._indices = indices


class TracePreservingOperation(Operation):
    """A general trace preserving operation.

    Parameters
    ----------
    transfer_matrix: array_like, optional
        Pauli transfer matrix of the operation.
    kraus: list of array_like, optional
        Kraus representation of the operation.
    basis: qs2.basis.PauliBasis
        Basis, in which the operation is provided.
        TODO: expand.
    """

    def __init__(self, *, ptm=None, kraus=None, basis=None):
        if ptm is not None and kraus is not None:
            raise ValueError(
                '`ptm` and `kraus` are exclusive parameters, '
                'specify only one of them.')
        if ptm is not None:
            _check_ptm_dims(ptm)
            ptm_dim_hilbert = ptm.shape[0]
            self._basis = basis or general(ptm_dim_hilbert)
            self._ptm = ptm
        elif kraus is not None:
            kraus_dim_hilbert = kraus.shape[-1]
            self._basis = basis or general(kraus_dim_hilbert)
            self._ptm = kraus_to_ptm(kraus, self._basis)
        else:
            raise ValueError('Specify either `transfer_matrix` or `kraus`.')

    def __call__(self, state, *indices):
        raise NotImplementedError()

    @property
    def n_qubits(self):
        return self._basis.num_subsystems


class Initialization(Operation):
    def __call__(self, state, *qubit_indices):
        pass

    @property
    def n_qubits(self):
        raise NotImplementedError()


class Measurement(Operation):
    def __call__(self, state, *qubit_indices):
        """Returns the result of the measurement"""
        pass

    @property
    def n_qubits(self):
        raise NotImplementedError()


class CombinedOperation(Operation):
    def __call__(self, state, *qubit_indices):
        pass

    @property
    def n_qubits(self):
        raise NotImplementedError()


def join(*operations):
    """Combines a list of operations into one operation.

    Parameters
    ----------
    op0, ..., opN: qs2.Operation or qs2.operation._DumbIndexedOperation
        Operations involved, in chronological order. If number of qubits in
        them is not the same everywhere, all of them must specify dumb
        indices of qubits they act onto with `Operation.at()` method.

    Returns
    -------
    qs2.Operation
        Combined operation. Exact type of the output operation may depend on
        input.
    """
    # input validation
    if len(operations) == 0:
        raise ValueError('Specify at least two operations to join.')
    op0 = operations[0]
    if isinstance(op0, Operation):
        cls = Operation
    elif isinstance(op0, _DumbIndexedOperation):
        cls = _DumbIndexedOperation
    else:
        raise ValueError(
            'Expected an operation, got {}'.format(type(op0)))
    for op in operations[1:]:
        if not isinstance(op, cls):
            raise ValueError(
                'Specify indices for all operations involved with '
                '`Operation.at()` method.')
    if isinstance(op0, Operation):
        for i, op in enumerate(operations[1:], start=1):
            if op0.n_qubits != op.n_qubits:
                raise ValueError(
                    'Numbers of qubits in operations 0 and {i} do not match:\n'
                    ' - operation 0 involves {n0} qubits\n'
                    ' - operation {i} involves {ni} qubits\n'
                    'Specify qubits to act on with `Operation.at()` method.'
                    .format(i=i, n0=op0.n_qubits, ni=op.n_qubits))

    # actual joining
    raise NotImplementedError()
