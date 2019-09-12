from .. import bases
import numpy as np


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

    def __init__(self, qubits, *, dim=2, pauli_vector_class=None,
                 pauli_vector=None):
        self.qubits = list(qubits)
        if pauli_vector is not None:
            self.pauli_vector = pauli_vector
        else:
            bases_ = (bases.general(dim).subbasis([0]),) * len(self.qubits)
            if pauli_vector_class is None:
                from ..pauli_vectors import Default
                self.pauli_vector = Default(bases_)
            else:
                self.pauli_vector = pauli_vector_class(bases_)

    def __copy__(self):
        return self.copy()

    def copy(self):
        return State(self.qubits, pauli_vector=self.pauli_vector.copy())

    def exp_value(self, operator):
        """

        Parameters
        ----------
        operator : numpy.array
            An operator (TODO: elaborate)

        Returns
        -------
        float
            Expectation value for each of the measurement operators, defined in
        """
        dm = self.pauli_vector.to_dm().reshape(self.pauli_vector.dim_hilbert*2)
        nq = len(self.qubits)
        in_idx = list(range(nq))
        out_idx = list(range(nq, 2*nq))
        return np.einsum(
            dm, in_idx + out_idx,
            operator, out_idx + in_idx
        )

    def partial_trace(self, *qubits):
        """Traces out all qubits, except provided, and returns the resulting
        state.

        Parameters
        ----------
        q0, q1, ... : str
            Names of qubits to preserve in the state.
        """
        return State(qubits, pauli_vector=self.pauli_vector.partial_trace(*[
            self.qubits.index(q) for q in qubits]))
