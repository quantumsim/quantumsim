import abc
import numpy as np
from functools import reduce
from ..state import State
from .common import kraus_to_ptm
from .common import _check_ptm_dims, _check_kraus_dims, _check_ptm_basis_consistency
from .common import convert_ptm_basis
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
    def num_qubits(self):
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

    def __init__(self, *, ptm=None, kraus=None, bases=None):
        if ptm is not None and kraus is not None:
            raise ValueError(
                '`ptm` and `kraus` are exclusive parameters, '
                'specify only one of them.')
        if ptm is not None:
            _check_ptm_dims(ptm)
            self._bases = bases or [general(ptm.shape[0])]
            _check_ptm_basis_consistency(ptm, self._bases)
            self._ptm = ptm
        elif kraus is not None:
            kraus = _check_kraus_dims(kraus)
            self._bases = bases or [general(kraus.shape[-1])]
            # basis check already done in the conversion
            self._ptm = kraus_to_ptm(kraus, self._bases)
        else:
            raise ValueError('Specify either `transfer_matrix` or `kraus`.')

        # To be removed in the future
        if self.num_qubits not in [1, 2]:
            raise NotImplementedError

    @property
    def num_qubits(self):
        return len(self._bases)

    @property
    def ptm(self):
        return self._ptm

    @property
    def bases(self):
        return self._bases

    def __call__(self, state, *qubit_indices):
        self._check_indices(qubit_indices)
        if self.num_qubits == 1:
            state.apply_single_qubit_ptm(*qubit_indices, self._ptm)
        elif self.num_qubits == 2:
            state.apply_two_qubit_ptm(*qubit_indices, self._ptm)
        else:
            raise NotImplementedError

    def to_bases(self, new_bases):
        if self._bases == new_bases:
            return self
        conv_ptm = convert_ptm_basis(self._ptm, self._bases, new_bases)
        return TracePreservingOperation(ptm=conv_ptm, bases=new_bases)

    def _check_indices(self, qubit_indices):
        if len(qubit_indices) != self.num_qubits:
            raise ValueError(
                'Incorrect number of indicies for a single qubit PTM')


class Initialization(Operation):
    def __call__(self, state, *qubit_indices):
        pass

    @property
    def num_qubits(self):
        raise NotImplementedError()


class Measurement(Operation):
    def __call__(self, state, *qubit_indices):
        """Returns the result of the measurement"""
        pass

    @property
    def num_qubits(self):
        raise NotImplementedError()


class CombinedOperation(Operation):
    def __init__(self, ptms, joint_bases):
        self._ptms = ptms
        self._bases = joint_bases

    def __call__(self, state, *qubit_indices):
        pass

    @property
    def n_qubits(self):
        return len(self._bases)


def join(*operations, out_bases=None):
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
    if len(operations) < 2:
        raise ValueError('Specify at least two operations to join.')

    init_op = operations[0]
    if isinstance(init_op, Operation):
        cls = Operation
    elif isinstance(init_op, _DumbIndexedOperation):
        cls = _DumbIndexedOperation
    else:
        raise ValueError(
            'Expected an operation, got {}'.format(type(init_op)))
    req_num_qubits = init_op.num_qubits
    req_bases = out_bases or init_op.bases

    if not all(isinstance(op_inst, cls) for op_inst in operations[1:]):
        raise ValueError(
            'Specify indices for all operations involved with '
            '`Operation.at()` method.')

    if cls is Operation:
        for i, op_inst in enumerate(operations[1:], start=1):
            if op_inst.num_qubits != req_num_qubits:
                raise ValueError(
                    'Numbers of qubits operation {i} acts on does not match the specified required number by the first operation:\n'
                    ' - Initial operation involves {n0} qubits\n'
                    ' - Operation {i} involves {ni} qubits\n'
                    'Specify the indices of the qubits that the operation acts on with the `Operation.at()` method.'
                    .format(i=i, n0=req_num_qubits, ni=op_inst.num_qubits))

        op_ptms = [op_inst.ptm if op_inst.bases == req_bases
                   else op_inst.to_bases(req_bases).ptm
                   for op_inst in operations]
        joint_ptm = reduce(np.dot, op_ptms)
        return TracePreservingOperation(ptm=joint_ptm, bases=req_bases)

    else:
        print()
        raise NotImplementedError()
    # for each operation: get basis, check that the basis match and if not convert them to the same base (specified by the basis of the first operation).
