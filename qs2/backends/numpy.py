import numpy as np
import pytools
import warnings
from .backend import DensityMatrixBase


class DensityMatrix(DensityMatrixBase):
    def __init__(self, dimensions, data=None):
        """A density matrix describing several subsystems with variable number
        of dimensions.

        Parameters
        ----------
        dimensions : list of int
            Dimensions of qubits in the system.

        data : numpy.ndarray or None.
            Must be of size (2**no_qubits, 2**no_qubits). Only upper triangle
            is relevant.  If data is `None`, create a new density matrix with
            all qubits in ground state.
        """
        raise NotImplementedError("Not yet unified with CUDA backend")
        self.dimensions = dimensions
        self.no_qubits = len(self.dimensions)
        self.size = pytools.product(self.dimensions)**2
        self.shape = [d**2 for d in dimensions]

        if len(dimensions) > 15:
            raise ValueError(
                "no_qubits=%d is way too many qubits, are you sure?" %
                len(dimensions))

        if isinstance(data, np.ndarray):
            single_tensors = [ptm.general_ptm_basis_vector(d)
                              for d in self.dimensions]
            assert data.size == self.size

            data = data.reshape(list(
                pytools.flatten([[d, d] for d in self.dimensions])))

            in_indices = list(
                reversed(range(self.no_qubits, 3 * self.no_qubits)))
            contraction_indices = [
                (i, i + self.no_qubits, i + 2 * self.no_qubits)
                for i in range(self.no_qubits)]
            out_indices = list(reversed(range(self.no_qubits)))

            transformation_tensors = list(
                zip(single_tensors, contraction_indices))
            transformation_tensors = pytools.flatten(transformation_tensors)

            self.dm = np.einsum(
                data, in_indices, *transformation_tensors, out_indices,
                optimize=True).real
        elif data is None:
            self.dm = np.zeros(self.shape)
            self.dm[tuple([0] * self.no_qubits)] = 1
        else:
            raise ValueError("Unknown type of `data`: {}".format(type(data)))

    def renormalize(self):
        self.dm = self.dm / self.trace()

    def copy(self):
        cp = self.__class__(self.dimensions)
        cp.dm = self.dm.copy()
        return cp

    def to_array(self):
        single_tensors = [ptm.general_ptm_basis_vector(d)
                          for d in self.dimensions]

        in_indices = list(reversed(range(self.no_qubits)))

        idx = [[i, 2 * self.no_qubits - i, 3 * self.no_qubits - i]
               for i in in_indices]

        transformation_tensors = list(zip(single_tensors, idx))
        transformation_tensors = pytools.flatten(transformation_tensors)

        density_matrix = np.einsum(
            self.dm, in_indices, *transformation_tensors, optimize=True)

        complex_dm_dimension = pytools.product(self.dimensions)
        density_matrix = density_matrix.reshape(
            (complex_dm_dimension, complex_dm_dimension))
        return density_matrix

    def get_diag(self):
        no_trace_tensors = []
        for d in self.dimensions:
            ntt = np.zeros((d**2, d))
            ntt[:d, :] = np.eye(d)
            no_trace_tensors.append(ntt)

        trace_argument = []
        for i, ntt in enumerate(no_trace_tensors):
            trace_argument.append(ntt)
            trace_argument.append([i, i + self.no_qubits])

        indices = list(reversed(range(self.no_qubits)))
        out_indices = list(reversed(range(self.no_qubits, 2 * self.no_qubits)))

        complex_dm_dimension = pytools.product(self.dimensions)
        return np.einsum(self.dm, indices, *trace_argument, out_indices,
                         optimize=True).reshape(complex_dm_dimension)

    def apply_two_qubit_ptm(self, bit0, bit1, two_ptm):

        d0 = self.shape[bit0]
        d1 = self.shape[bit1]

        two_ptm = two_ptm.reshape((d1, d0, d1, d0))

        dummy_idx0, dummy_idx1 = self.no_qubits, self.no_qubits + 1
        out_indices = list(reversed(range(self.no_qubits)))
        in_indices = list(reversed(range(self.no_qubits)))
        in_indices[self.no_qubits - bit0 - 1] = dummy_idx0
        in_indices[self.no_qubits - bit1 - 1] = dummy_idx1
        two_ptm_indices = [
            bit1, bit0,
            dummy_idx1, dummy_idx0
        ]
        self.dm = np.einsum(
            self.dm, in_indices, two_ptm, two_ptm_indices, out_indices,
            optimize=True)

    def apply_single_qubit_ptm(self, qubit, ptm, basis_out=None):
        self._validate_bit(bit, 'bit')
        dim = self.shape[bit]
        self._validate_ptm_shape(ptm, (dim, dim), 'ptm')

        dummy_idx = self.no_qubits
        out_indices = list(reversed(range(self.no_qubits)))
        in_indices = list(reversed(range(self.no_qubits)))
        in_indices[self.no_qubits - bit - 1] = dummy_idx
        ptm_indices = [bit, dummy_idx]
        self.dm = np.einsum(self.dm, in_indices, ptm,
                            ptm_indices, out_indices, optimize=True)

    def add_ancilla(self, anc_st, anc_dim):
        anc_dm = np.zeros(anc_dim**2)
        anc_dm[anc_st] = 1
        self.dm = np.einsum(
            anc_dm, [0], self.dm, list(range(1, self.no_qubits + 1)),
            optimize=True)
        self.dimensions.insert(0, anc_dim)
        self.shape = [d**2 for d in self.dimensions]
        self.no_qubits = len(self.dimensions)

    def partial_trace(self, bit):
        if bit >= self.no_qubits:
            raise ValueError("Bit '{}' does not exist".format(bit))

        trace_argument = []
        for i, d in enumerate(self.dimensions):
            if i == bit:
                ntt = np.zeros((d, d**2))
                ntt[:, :d] = np.eye(d)
                trace_argument.append(ntt)
                trace_argument.append([self.no_qubits + 1, i])
            else:
                tt = np.zeros(d**2)
                tt[:d] = 1
                trace_argument.append(tt)
                trace_argument.append([i])

        indices = list(reversed(range(self.no_qubits)))

        return np.einsum(self.dm, indices, *trace_argument, optimize=True)

    def trace(self):

        trace_argument = []
        for i, d in enumerate(self.dimensions):
            tt = np.zeros(d**2)
            tt[:d] = 1
            trace_argument.append(tt)
            trace_argument.append([i])

        return np.einsum(self.dm, list(range(self.no_qubits)), *trace_argument,
                         optimize=True)

    def project(self, qubit, state):
        self._validate_qubit(qubit, 'bit')

        # the behaviour is a bit weird: swap the MSB to bit and then project
        # out the highest one!
        dim = self.shape[qubit]
        projector = np.zeros(dim)
        projector[state] = 1

        in_indices = list(reversed(range(self.no_qubits)))
        projector_indices = [qubit]
        out_indices = list(reversed(range(self.no_qubits - 1)))
        if qubit != self.no_qubits - 1:
            out_indices[-qubit - 1] = self.no_qubits - 1

        self.dm = np.einsum(self.dm, in_indices, projector,
                            projector_indices, out_indices, optimize=True)

        self.dimensions = self.dimensions[1:]
        self.no_qubits = len(self.dimensions)


