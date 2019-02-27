from .. import bases
import importlib
import numpy as np


class State:
    """

    Parameters
    ----------
    basis : None, qs2.bases.PauliBasis or list of qs2.bases.PauliBasis
        A superbasis, used for the computation. If None, Quantumsim desides.
    data : qs2.backends.State
        State of the system.
    """
    def __init__(self, data):
        self.dm = data

    @property
    def basis(self):
        return self.dm.bases

    @classmethod
    def from_pv(cls, pv, basis, *, backend_cls=None):
        if basis is None:
            basis = bases.general(2)
        if not hasattr(basis, '__iter__'):
            basis = [basis] * len(pv.shape)
        if backend_cls is None:
            mod = importlib.import_module('qs2.backends')
            backend_cls = mod.DensityMatrix
        data = backend_cls(basis, expansion=pv)
        return cls(data)

    def probability(self, *qubits):
        """Returns a probability of the measurement, not actually performing it.

        Parameters
        ----------
        i0, i1, ..., iN: int
            Indices of qubits of interest in the state. If none of them are
            specified, all of qubits in the state are returned.
        axis: str
            Basis, in which a measurement should be performed. Default is `z`.

        Returns
        -------
        list of tuples
            Probabilities for qubits to stay in a certain state (0, 1, 2, ...),
            dependent on the dimensionality of qubits' basis.
        """
        raise NotImplementedError()

    def project(self, qubit, comp_state):
        state_index = self.basis[qubit].computational_basis_indices[comp_state]
        if state_index is None:
            raise NotImplementedError(
                'Projected state is not in the computational basis indices; '
                'this is not supported.'
            )
        self.dm.project(qubit, state_index)

        self.basis[qubit] = self.basis[qubit].subbasis([state_index])


