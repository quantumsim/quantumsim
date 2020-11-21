import warnings

import numpy as np
from .pauli_vector import PauliVectorBase, prod


class PauliVectorNumpy(PauliVectorBase):
    def __init__(self, bases, pv=None, *, force=False):
        """A density matrix describing several subsystems with variable number
        of dimensions.

        Parameters
        ----------
        bases : list of quantumsim.bases.PauliBasis
            Dimensions of qubits in the system.

        pv : array or None.
            Must be of size (2**no_qubits, 2**no_qubits). Only upper triangle
            is relevant.  If data is `None`, create a new density matrix with
            all qubits in ground state.
        """
        super().__init__(bases, pv, force=force)
        if pv is not None:
            if self.dim_pauli != pv.shape:
                raise ValueError(
                    '`bases` Pauli dimensionality should be the same as the '
                    'shape of `data` array.\n'
                    ' - bases shapes: {}\n - data shape: {}'
                    .format(self.dim_pauli, pv.shape))
            if pv.dtype not in (np.float16, np.float32, np.float64):
                raise ValueError(
                    '`pv` must have floating point data type, got {}'
                    .format(pv.dtype)
                )

        if isinstance(pv, np.ndarray):
            self._data = pv
        elif pv is None:
            self._data = np.zeros(self.dim_pauli)
            self._data[tuple([0] * self.n_qubits)] = 1
        else:
            raise ValueError(
                "`pv` should be Numpy array or None, got type `{}`"
                .format(type(pv)))

    def to_pv(self):
        return self._data

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
            optimize='greedy')

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
        complex_dm_dimension = prod(self.dim_hilbert)
        return np.einsum(self._data, indices, *trace_argument, out_indices,
                         optimize='greedy').real.reshape(complex_dm_dimension)

    def trace(self):
        # TODO: can be made more effective
        return np.sum(self.diagonal())

    def partial_trace(self, *qubits):
        for q in qubits:
            self._validate_qubit(q, 'qubit')
        einsum_args = [self._data, list(range(self.n_qubits))]
        for i, b in enumerate(self.bases):
            if i not in qubits:
                einsum_args.append(b.vectors)
                einsum_args.append([i, self.n_qubits+i, self.n_qubits+i])
        traced_dm = np.einsum(*einsum_args, optimize='greedy').real
        return self.__class__([self.bases[q] for q in qubits], traced_dm)

    def meas_prob(self, qubit):
        self._validate_qubit(qubit, 'qubit')
        einsum_args = [self._data, list(range(self.n_qubits))]
        for i, b in enumerate(self.bases):
            einsum_args.append(b.vectors)
            einsum_args.append([i, self.n_qubits+i, self.n_qubits+i])
        einsum_args.append([self.n_qubits + qubit])
        try:
            return np.einsum(*einsum_args, optimize='greedy').real
        except Exception:
            raise

    def renormalize(self):
        tr = self.trace()
        if tr > 1e-8:
            self._data *= self.trace() ** -1
        else:
            warnings.warn(
                "Density matrix trace is 0; likely your further computation "
                "will fail. Have you projected DM on a state with zero weight?")
        return tr

    def copy(self):
        return self.from_pv(self.to_pv().copy(), self.bases)

