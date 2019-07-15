import abc
import pytools
from quantumsim.algebra.algebra import dm_to_pv, pv_to_dm


class PauliVectorBase(metaclass=abc.ABCMeta):
    """A metaclass, that defines standard interface for Quantumsim density
    matrix backend.

    Every instance, that implements the interface of this class, should call
    `super().__init__` in the beginning of its execution, because a lot of
    sanity checks are done here.

    Parameters
    ----------
    bases : list of quantumsim.bases.PauliBasis
        A descrption of the basis for the subsystems.
    pv : array or None
        Pauli vector, that represents the density matrix in the selected
        bases. If `None`, density matrix is initialized in
        :math:`\\left| 0 \\cdots 0 \\right\\rangle` state.
    force : bool
        By default creation of too large density matrix (more than
        :math:`2^22` elements currently) is not allowed. Set this to `True`
        if you know what you are doing.
    """
    _size_max = 2**22

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def __init__(self, bases, pv=None, *, force=False):
        self.bases = list(bases)
        if self.size > self._size_max and not force:
            raise ValueError(
                'Density matrix of the system is going to have {} items. It '
                'is probably too much. If you know what you are doing, '
                'pass `force=True` argument to the constructor.')

    @classmethod
    def from_pv(cls, pv, bases, *, force=False):
        return cls(bases, pv, force=force)

    @abc.abstractmethod
    def to_pv(self):
        """Get data in a form of Numpy array"""
        pass

    @classmethod
    def from_dm(cls, dm, bases, *, force=False):
        if not hasattr(bases, '__iter__'):
            n_qubits = len(dm) // bases.dim_hilbert
            bases = [bases] * n_qubits
        return cls(bases, dm_to_pv(dm, bases), force=force)

    def to_dm(self):
        return pv_to_dm(self.to_pv(), self.bases)

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
    def apply_ptm(self, operation, *qubits):
        pass

    @abc.abstractmethod
    def diagonal(self, *, get_data=True):
        pass

    @abc.abstractmethod
    def trace(self):
        pass

    @abc.abstractmethod
    def partial_trace(self, *qubits):
        pass

    @abc.abstractmethod
    def meas_prob(self, qubit):
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
