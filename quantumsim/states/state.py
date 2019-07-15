from .. import bases


class State:
    """
    Parameters
    ----------
    qubits : list of str
        Names of qubits in state
    dim : int
        Hilbert dimensionality of a single-qubit subspace
    pauli_vector_class : class or None
        A class to store the system state. Must be a derivative of
        :class:`quantumsim.pauli_vectors.pauli_vector.PauliVectorBase`.
    """
    def __init__(self, qubits, *, dim=2, pauli_vector_class=None):
        self._qubits = qubits
        bases_ = (bases.general(dim).subbasis([0]),) * len(self._qubits)
        if pauli_vector_class is None:
            from ..pauli_vectors import Default
            self._pauli_vector = Default(bases_)
        else:
            self._pauli_vector = pauli_vector_class(bases_)
