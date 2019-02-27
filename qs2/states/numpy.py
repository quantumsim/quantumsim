import warnings

import numpy as np
import pytools
from .backend import State
from ..operations.operation import PTMOperation


class DensityMatrix(State):
    def __init__(self, bases, expansion=None, *, force=False):
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
        self._bases = list(bases)
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

        if isinstance(expansion, np.ndarray):
            self._data = expansion
        elif expansion is None:
            self._data = np.zeros(self.dim_pauli)
            self._data[tuple([0] * self.n_qubits)] = 1
        else:
            raise ValueError(
                "`expansion` should be Numpy array or None, got type `{}`"
                .format(type(expansion)))

    @property
    def bases(self):
        return self._bases

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
        if len(ptm.shape) != 2 * len(qubits):
            raise ValueError(
                '{}-qubit PTM must have {} dimensions, got {}'
                .format(len(qubits), 2*len(qubits), len(ptm.shape)))
        dm_in_idx = list(range(self.n_qubits))
        ptm_in_idx = list(qubits)
        ptm_out_idx = list(range(self.n_qubits, self.n_qubits + len(
            qubits)))
        dm_out_idx = list(dm_in_idx)
        for i_in, i_out in zip(ptm_in_idx, ptm_out_idx):
            dm_out_idx[i_in] = i_out
        self._data = np.einsum(
            self._data, dm_in_idx, ptm, ptm_out_idx + ptm_in_idx, dm_out_idx,
            optimize=True)

    def add_qubit(self, basis, classical_state):
        self._data = np.einsum(
            basis.computational_basis_vectors[classical_state], [0],
            self._data, list(range(1, self.n_qubits + 1)),
            optimize=True)
        self.bases.insert(0, basis)

    def partial_trace(self, qubit):
        self._validate_qubit(qubit, 'qubit')
        einsum_args = [self._data, list(range(self.n_qubits))]
        for i, b in enumerate(self.bases):
            if i != qubit:
                einsum_args.append(b.vectors)
                einsum_args.append([i, self.n_qubits+i, self.n_qubits+i])
        traced_dm = np.einsum(*einsum_args, optimize=True).real
        return self.__class__([self.bases[qubit]], traced_dm)

    def meas_prob(self, qubit):
        self._validate_qubit(qubit, 'qubit')
        einsum_args = [self._data, list(range(self.n_qubits))]
        for i, b in enumerate(self.bases):
            einsum_args.append(b.vectors)
            einsum_args.append([i, self.n_qubits+i, self.n_qubits+i])
        einsum_args.append([self.n_qubits + qubit])
        try:
            return np.einsum(*einsum_args, optimize=True).real
        except Exception:
            raise

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
