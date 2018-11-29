import numpy as np
import pytools
from .backend import DensityMatrixBase


class DensityMatrix(DensityMatrixBase):
    def __init__(self, bases, expansion=None):
        """A density matrix describing several subsystems with variable number
        of dimensions.

        Parameters
        ----------
        bases : list of qs2.bases.PauliBasis
            Dimensions of qubits in the system.

        expansion : numpy.ndarray or None.
            Must be of size (2**no_qubits, 2**no_qubits). Only upper triangle
            is relevant.  If data is `None`, create a new density matrix with
            all qubits in ground state.
        """
        super().__init__(bases, expansion)

        if isinstance(expansion, np.ndarray):
            self._data = expansion
        elif expansion is None:
            self._data = np.zeros(self.shape)
            self._data[tuple([0] * self.n_qubits)] = 1
        else:
            raise ValueError(
                "`expansion` should be Numpy array or None, got type `{}`"
                .format(type(expansion)))

    def expansion(self):
        return self._data

    def renormalize(self):
        self._data /= self.trace()

    def copy(self):
        cp = self.__class__(self.dimensions)
        cp._data = self._data.copy()
        return cp

    def diagonal(self, *, get_data=True):
        # FIXME: Works only for qs2.basis.general
        no_trace_tensors = []
        for d in self.dimensions:
            ntt = np.zeros((d**2, d))
            ntt[:d, :] = np.eye(d)
            no_trace_tensors.append(ntt)

        trace_argument = []
        n_qubits = self.n_qubits
        for i, ntt in enumerate(no_trace_tensors):
            trace_argument.append(ntt)
            trace_argument.append([i, i + n_qubits])

        indices = list(reversed(range(n_qubits)))
        out_indices = list(reversed(range(n_qubits, 2 * n_qubits)))

        complex_dm_dimension = pytools.product(self.dimensions)
        return np.einsum(self._data, indices, *trace_argument, out_indices,
                         optimize=True).reshape(complex_dm_dimension)

    def apply_two_qubit_ptm(self, qubit0, qubit1, two_ptm, basis_out=None):
        d0 = self.shape[qubit0]
        d1 = self.shape[qubit1]

        two_ptm = two_ptm.reshape((d1, d0, d1, d0))

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

        if basis_out is not None:
            self.bases[qubit0] = basis_out[0]
            self.bases[qubit1] = basis_out[1]

    def apply_single_qubit_ptm(self, qubit, ptm, basis_out=None):
        self._validate_qubit(qubit, 'bit')
        dim = self.shape[qubit]
        self._validate_ptm_shape(ptm, (dim, dim), 'ptm')

        n_qubits = self.n_qubits
        dummy_idx = n_qubits
        out_indices = list(reversed(range(n_qubits)))
        in_indices = list(reversed(range(n_qubits)))
        in_indices[n_qubits - qubit - 1] = dummy_idx
        ptm_indices = [qubit, dummy_idx]
        self._data = np.einsum(self._data, in_indices, ptm,
                               ptm_indices, out_indices, optimize=True)

        if basis_out is not None:
            self.bases[qubit] = basis_out

    def add_ancilla(self, basis, state):
        raise NotImplementedError("TODO for Numpy backend")
        # self.dm = np.einsum(
        #     anc_dm, [0], self.dm, list(range(1, self.no_qubits + 1)),
        #     optimize=True)
        # self.bases.insert(0, basis)

    def partial_trace(self, qubit):
        self._validate_qubit(qubit, 'qubit')

        trace_argument = []
        for i, d in enumerate(self.dimensions):
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

        trace_argument = []
        for i, d in enumerate(self.dimensions):
            tt = np.zeros(d**2)
            tt[:d] = 1
            trace_argument.append(tt)
            trace_argument.append([i])

        return np.einsum(self._data, list(range(self.n_qubits)),
                         *trace_argument, optimize=True)

    def project(self, qubit, state):
        self._validate_qubit(qubit, 'qubit')

        # the behaviour is a bit weird: swap the MSB to bit and then project
        # out the highest one!
        dim = self.shape[qubit]
        projector = np.zeros(dim)
        projector[state] = 1

        n_qubits = self.n_qubits
        in_indices = list(reversed(range(n_qubits)))
        projector_indices = [qubit]
        out_indices = list(reversed(range(n_qubits - 1)))
        if qubit != n_qubits - 1:
            out_indices[-qubit - 1] = n_qubits - 1

        self._data = np.einsum(self._data, in_indices, projector,
                               projector_indices, out_indices, optimize=True)
