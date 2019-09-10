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
        self.qubits = list(qubits)
        bases_ = (bases.general(dim).subbasis([0]),) * len(self.qubits)
        if pauli_vector_class is None:
            from ..pauli_vectors import Default
            self.pauli_vector = Default(bases_)
        else:
            self.pauli_vector = pauli_vector_class(bases_)

    def exp_values(self, measurements):
        """

        Parameters
        ----------
        measurements : list of dict
            List of measurement dictionaries, containing 'X', 'Y' or 'Z' for
            each non-trivial qubit label.

        Returns
        -------
        list of float
            Expectation value for each of the measurement operators, defined in
        """
        raise NotImplementedError

    def partial_trace(self, *qubits):
        """Traces out all qubits, except provided, and returns the resulting
        state.

        Parameters
        ----------
        q0, q1, ... : str
            Names of qubits to preserve in the state.
        """
        out = State(qubits, dim=self.pauli_vector.dim_hilbert[0])
        out.pauli_vector = self.pauli_vector.partial_trace(*[
            self.qubits.index(q) for q in qubits])
        return out
