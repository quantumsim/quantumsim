import warnings

import numpy as np
import pytools
from .backend import DensityMatrixBase
from ..operations.operation import PTMOperation


class DensityMatrix(DensityMatrixBase):
    def __init__(self, bases, expansion=None):
        """A density matrix describing several subsystems with variable number
        of dimensions.

        Parameters
        ----------
        bases : tuple of qs2.bases.PauliBasis
            Dimensions of qubits in the system.

        expansion : array or None.
            Must be of size (2**no_qubits, 2**no_qubits). Only upper triangle
            is relevant.  If data is `None`, create a new density matrix with
            all qubits in ground state.
        """
        super().__init__(bases, expansion)

        if isinstance(expansion, np.ndarray):
            self._data = expansion
        elif expansion is None:
            self._data = np.zeros(self.dim_pauli)
            self._data[tuple([0] * self.n_qubits)] = 1
        else:
            raise ValueError(
                "`expansion` should be Numpy array or None, got type `{}`"
                .format(type(expansion)))

    def expansion(self):
        return self._data

    def renormalize(self):
        tr = self.trace()
        if tr > 1e-8:
            self._data /= self.trace()
        else:
            warnings.warn(
                "Density matrix trace is 0; likely your further computation "
                "will fail. Have you projected DM on a state with zero weight?")

    def copy(self):
        cp = self.__class__(self.dim_hilbert)
        cp._data = self._data.copy()
        return cp

    def diagonal(self, *, get_data=True):
        no_trace_tensors = [basis.computational_basis_vectors
                            for basis in self.bases]

        trace_argument = []
        n_qubits = self.n_qubits
        for i, ntt in enumerate(no_trace_tensors):
            trace_argument.append(ntt)
            trace_argument.append([i + n_qubits, i])

        indices = list(range(n_qubits))
        out_indices = list(range(n_qubits, 2 * n_qubits))
        complex_dm_dimension = pytools.product(self.dim_hilbert)
        return np.einsum(self._data, indices, *trace_argument, out_indices,
                         optimize=True).reshape(complex_dm_dimension)

    def apply_ptm(self, ptm, *qubits):
        if len(qubits) == 1:
            self._apply_single_qubit_ptm(qubits[0], ptm)
        elif len(qubits) == 2:
            self._apply_two_qubit_ptm(qubits[0], qubits[1], ptm)
        else:
            raise NotImplementedError('Applying {}-qubit PTM is not '
                                      'implemented in the active backend.')

    def _apply_two_qubit_ptm(self, qubit0, qubit1, two_ptm):
        n_qubits = self.n_qubits
        dummy_idx0, dummy_idx1 = n_qubits, n_qubits + 1
        out_indices = list(reversed(range(n_qubits)))
        in_indices = list(reversed(range(n_qubits)))
        in_indices[n_qubits - qubit0 - 1] = dummy_idx0
        in_indices[n_qubits - qubit1 - 1] = dummy_idx1
        two_ptm_indices = [
            qubit1, qubit0,
            dummy_idx1, dummy_idx0
        ]
        self._data = np.einsum(
            self._data, in_indices, two_ptm, two_ptm_indices, out_indices,
            optimize=True)

    def _apply_single_qubit_ptm(self, qubit, ptm):
        self._validate_qubit(qubit, 'bit')
        dim = self.dim_pauli[qubit]

        n_qubits = self.n_qubits
        dummy_idx = n_qubits
        out_indices = list(reversed(range(n_qubits)))
        in_indices = list(reversed(range(n_qubits)))
        in_indices[n_qubits - qubit - 1] = dummy_idx
        ptm_indices = [qubit, dummy_idx]
        self._data = np.einsum(self._data, in_indices, ptm,
                               ptm_indices, out_indices, optimize=True)

    def add_qubit(self, basis, classical_state):
        self._data = np.einsum(
            basis.computational_basis_vectors[classical_state], [0],
            self._data, list(range(1, self.n_qubits + 1)),
            optimize=True)
        self.bases.insert(0, basis)

    def partial_trace(self, qubit):
        self._validate_qubit(qubit, 'qubit')

        trace_argument = []
        for i, d in enumerate(self.dim_hilbert):
            if i == qubit:
                ntt = np.zeros((d, d**2))
                ntt[:, :d] = np.eye(d)
                trace_argument.append(ntt)
                trace_argument.append([self.n_qubits + 1, i])
            else:
                tt = np.zeros(d**2)
                tt[:d] = 1
                trace_argument.append(tt)
                trace_argument.append([i])

        indices = list(reversed(range(self.n_qubits)))

        return np.einsum(self._data, indices, *trace_argument, optimize=True)

    def trace(self):
        # TODO: can be made more effective
        return np.sum(self.diagonal())

    def project(self, qubit, state):
        self._validate_qubit(qubit, 'bit')
        target_qubit_state_index = self.bases[qubit] \
            .computational_basis_indices[state]
        if target_qubit_state_index is None:
            raise RuntimeError(
                'Projected state is not in the computational basis indices; '
                'this is not supported.'
            )

        idx = [tuple(range(dp)) for dp in self.dim_pauli]
        idx[qubit] = (target_qubit_state_index,)
        self._data = self._data[np.ix_(*idx)]
        self.bases[qubit] = self.bases[qubit].subbasis([state])
