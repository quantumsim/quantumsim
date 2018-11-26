import abc


class DensityMatrixBase(metaclass=abc.ABCMeta):
    """A metaclass, that defines standard interface for Quantumsim backend."""

    def __init__(self, bases, data=None):
        self.bases = bases

    @property
    def n_qubits(self):
        return len(self.bases)

    @property
    def shape(self):
        return tuple([pb.dim_pauli for pb in self.bases])

    @abc.abstractmethod
    def apply_single_qubit_ptm(self, qubit, ptm, basis_out=None):
        pass

    @abc.abstractmethod
    def apply_two_qubit_ptm(self, qubit0, qubit1, ptm, basis_out=None):
        pass

    @abc.abstractmethod
    def get_diag(self, target_array=None, get_data=True, flatten=True):
        pass

    @abc.abstractmethod
    def trace(self):
        pass

    @abc.abstractmethod
    def partial_trace(self, qubit):
        pass

    @abc.abstractmethod
    def project(self, qubit, state):
        pass

    @abc.abstractmethod
    def renormalize(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def to_array(self):
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


