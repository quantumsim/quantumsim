import abc
from ..state import State
from .common import kraus_to_transfer_matrix


class Operation(metaclass=abc.ABCMeta):
    """A metaclass for all gates.

    Every gate has to implement call method, that takes a
    :class:`qs2.state.State` object and modifies it inline. This method may
    return nothing or result of a measurement, if it is a gate.
    """

    def __init__(self, indices=None):
        if indices is None:
            self._indices = tuple(range(self.n_qubits))
        else:
            self._indices = indices

    @abc.abstractmethod
    def __call__(self, state, *qubit_indices):
        """Applies the operation inline (modifying the state) to the state
        to certain qubits. Number of qubit indices should be aligned with a
        dimensionality of the operation.
        """
        pass

    def __matmul__(self, other):
        pass

    def at(self, *indices):
        copy = self.copy()
        copy._indices = indices
        return copy

    @property
    @abc.abstractmethod
    def n_qubits(self):
        pass

    def copy(self):
        raise NotImplementedError()


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

    def __init__(self, *, transfer_matrix=None, kraus=None, basis=None, indices=None, double_kraus=False):
        if transfer_matrix and kraus:
            raise ValueError(
                '`transfer_matrix` and `kraus` are exclusive parameters, '
                'specify only one of them.')
        if transfer_matrix is not None:
            self._transfer_matrix = transfer_matrix
        elif kraus is not None:
            self._transfer_matrix =
            kraus_to_transfer_matrix(
                kraus=kraus, basis=basis, double_kraus=double_kraus)
        else:
            raise ValueError('Specify either `transfer_matrix` or `kraus`.')
        self._basis = basis

        super().__init__(indices)

    def __call__(self, state, *indices):
        raise NotImplementedError()


class Initialization(Operation):
    def __call__(self, state, *qubit_indices):
        pass


class Measurement(Operation):
    def __call__(self, state, *qubit_indices):
        """Returns the result of the measurement"""
        pass


class CombinedOperation(Operation):
    def __call__(self, state, *qubit_indices):
        pass
