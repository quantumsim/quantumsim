import abc
import pytools
import numpy as np


class DensityMatrixBase(metaclass=abc.ABCMeta):
    """A metaclass, that defines standard interface for Quantumsim density
    matrix backend.

    Every instance, that implements the interface of this class, should call
    `super().__init__` in the beginning of its execution, because a lot of
    sanity checks are done here.

    Parameters
    ----------
    bases : list of qs2.bases.PauliBasis
        A descrption of the basis for the subsystems.
    expansion : numpy.ndarray, pycuda.gpuarray.GPUArray  or None
        Expansion of density matrix in the selected bases. If None, density
        matrix is initialized in :math:`\\left| 0 \\cdots 0 \\right\\rangle`
        state. Some sanity checks are done in the abstract class constructor,
        but handling the data is a task of implementation.
    force : bool
        By default creation of too large density matrix (more than
        :math:`2^22` elements currently) is not allowed. Set this to `True`
        if you know what you are doing.
    """
    _size_max = 2**22

    def __init__(self, bases, expansion=None, *, force=False):
        # FIXME: We do not support anything except general basis yet
        for b in bases:
            for k, v in b.computational_basis_indices.items():
                if not k == v:
                    raise NotImplementedError(
                        "Currently backends implicitly assume that basis is "
                        "always `qs2.basis.general()`. This should be fixed "
                        "as soon as possible."
                    )

        self.bases = bases

        if self.size > self._size_max and not force:
            raise ValueError(
                'Density matrix of the system is going to have {} items. It '
                'is probably too much. If you know what you are doing, '
                'pass `force=True` argument to the constructor.')

        if expansion is not None:
            if self.dim_pauli != expansion.shape:
                raise ValueError(
                    '`bases` Pauli dimensionality should be the same as the '
                    'shape of `data` array.\n'
                    ' - bases shapes: {}\n - data shape: {}'
                    .format(self.dim_pauli, expansion.shape))
            if expansion.dtype not in (np.float16, np.float32, np.float64):
                raise ValueError(
                    '`expansion` must have floating point data type, got {}'
                    .format(expansion.dtype)
                )

    @property
    def n_qubits(self):
        return len(self.bases)

    @property
    def dim_hilbert(self):
        return tuple((b.dim_hilbert for b in self.bases))

    @property
    def size(self):
        return pytools.product(self.dim_hilbert) ** 2

    @property
    def dim_pauli(self):
        return tuple([pb.dim_pauli for pb in self.bases])

    @abc.abstractmethod
    def expansion(self):
        """Get data in a form of Numpy array"""
        pass

    @abc.abstractmethod
    def apply_single_qubit_ptm(self, qubit, ptm, basis_out=None):
        # TODO Check exact signature of basis_out, I assume this does not
        # make any impact on PTM application
        pass

    @abc.abstractmethod
    def apply_two_qubit_ptm(self, qubit0, qubit1, ptm, basis_out=None):
        # TODO Check exact signature of basis_out, I assume this does not
        # make any impact on PTM application
        pass

    @abc.abstractmethod
    def diagonal(self, *, get_data=True):
        pass

    @abc.abstractmethod
    def trace(self):
        pass

    @abc.abstractmethod
    def partial_trace(self, qubit):
        pass

    @abc.abstractmethod
    def add_qubit(self, basis, classical_state):
        """Add a qubit to the density matrix and initialize it.

        Parameters
        ----------
        basis: qs2.bases.PauliBasis
            Initial basis for a qubit
        classical_state: int
            State of the qubit
        """
        pass

    @abc.abstractmethod
    def project(self, qubit, state):
        """Project a qubit to a state."""
        pass

    @abc.abstractmethod
    def renormalize(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    def _validate_qubit(self, number, name):
        if number < 0 or number >= self.n_qubits:
            raise ValueError(
                "`{name}` number {n} does not exist in the system, "
                "it contains {n_qubits} qubits in total."
                .format(name=name, n=number, n_qubits=self.n_qubits))

    # noinspection PyMethodMayBeStatic
    def _validate_ptm_shape(self, ptm, target_shape, name):
        if ptm.shape != target_shape:
            raise ValueError(
                "`{name}` shape must be {target_shape}, got {real_shape}"
                .format(name=name,
                        target_shape=target_shape,
                        real_shape=ptm.shape))
