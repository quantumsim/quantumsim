import abc
import pytools


class DensityMatrixBase(metaclass=abc.ABCMeta):
    """A metaclass, that defines standard interface for Quantumsim backend.
    """
    _size_max = 2**22

    def __init__(self, bases, expansion=None, *, force=False):
        self.bases = bases

        if self.size > self._size_max and not force:
            raise ValueError(
                'Density matrix of the system is going to have {} items. It '
                'is probably too much. If you know what you are doing, '
                'pass `force=True` argument to the constructor.')

        if expansion is not None and self.shape != expansion.shape:
            raise ValueError(
                '`bases` Pauli dimensionality should be the same as the shape '
                'of `data` array.\n - bases shapes: {}\n - data shape: {}'
                .format(self.shape, expansion.shape))

    @property
    def n_qubits(self):
        return len(self.bases)

    @property
    def dimensions(self):
        return tuple((b.dim_hilbert for b in self.bases))

    @property
    def size(self):
        return pytools.product(self.dimensions)**2

    @property
    def shape(self):
        return tuple([pb.dim_pauli for pb in self.bases])

    @property
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


