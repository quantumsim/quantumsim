import numpy as np
import pytools

from . import ptm
import warnings


class DensityNP:
    def __init__(self, no_qubits, data=None):

        if no_qubits > 15:
            raise ValueError(
                "no_qubits=%d is way too many qubits, are you sure?" %
                no_qubits)

        self.no_qubits = no_qubits
        self.shape = [4] * no_qubits

        if isinstance(data, np.ndarray):
            single_tensor = ptm.single_tensor
            assert data.size == 4**self.no_qubits

            data = data.reshape((2, 2) * self.no_qubits)

            in_indices = list(reversed(range(self.no_qubits, 3*self.no_qubits)))
            contraction_indices = [(i, i + self.no_qubits, i + 2*self.no_qubits) for i in range(self.no_qubits)]
            out_indices = list(reversed(range(self.no_qubits)))

            transformation_tensors = list(zip([single_tensor]*self.no_qubits, contraction_indices))
            transformation_tensors = pytools.flatten(transformation_tensors)

            self.dm = np.einsum(data, in_indices, *transformation_tensors, out_indices, optimize=True).real
        elif data is None:
            self.dm = np.zeros(self.shape)
            self.dm[tuple([0] * self.no_qubits)] = 1
        else:
            raise ValueError("type of data not understood")

    def renormalize(self):
        self.dm = self.dm / self.trace()

    def copy(self):
        cp = DensityNP(no_qubits=self.no_qubits)
        cp.dm = self.dm.copy()
        return cp

    def to_array(self):
        single_tensor = ptm.single_tensor

        in_indices = list(reversed(range(self.no_qubits)))

        idx = [[i, 2*self.no_qubits - i, 3*self.no_qubits - i]
               for i in in_indices]

        transformation_tensors = list(zip([single_tensor]*self.no_qubits, idx))
        transformation_tensors = pytools.flatten(transformation_tensors)

        density_matrix = np.einsum(self.dm, in_indices, *transformation_tensors, optimize=True)
        density_matrix = density_matrix.reshape((2**self.no_qubits, 2**self.no_qubits))
        return density_matrix

    def get_diag(self):

        no_trace_tensor = np.array([[1, 0, 0, 0], [0, 0, 0, 1]]).T

        trace_argument = []
        for i in range(self.no_qubits):
            trace_argument.append(no_trace_tensor)
            trace_argument.append([i, i + self.no_qubits])

        indices = list(reversed(range(self.no_qubits)))
        out_indices = list(reversed(range(self.no_qubits, 2*self.no_qubits)))

        return np.einsum(self.dm, indices, *trace_argument, out_indices, optimize=True).reshape(2**self.no_qubits)

    def apply_two_ptm(self, bit0, bit1, two_ptm):
        two_ptm = two_ptm.reshape((4, 4, 4, 4))
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
            self.dm, in_indices, two_ptm, two_ptm_indices, out_indices, optimize=True)

    def apply_ptm(self, bit, one_ptm):
        assert bit < self.no_qubits

        dummy_idx = self.no_qubits
        out_indices = list(reversed(range(self.no_qubits)))
        in_indices = list(reversed(range(self.no_qubits)))
        in_indices[self.no_qubits - bit - 1] = dummy_idx
        ptm_indices = [bit, dummy_idx]
        self.dm = np.einsum(self.dm, in_indices, one_ptm, ptm_indices, out_indices, optimize=True)

    def add_ancilla(self, anc_st):
        anc_dm = np.zeros(4)
        if anc_st == 1:
            anc_dm[3] = 1
        else:
            anc_dm[0] = 1
        self.dm = np.einsum(
            anc_dm, [0], self.dm, list(range(1, self.no_qubits + 1)), optimize=True)
        self.no_qubits = len(self.dm.shape)

    def partial_trace(self, bit):
        if bit >= self.no_qubits:
            raise ValueError("bit does not exist")

        trace_tensor = np.array([1, 0, 0, 1])
        no_trace_tensor = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])

        trace_argument = []
        for i in range(self.no_qubits):
            if i == bit:
                trace_argument.append(no_trace_tensor)
                trace_argument.append([self.no_qubits + 1, i])
            else:
                trace_argument.append(trace_tensor)
                trace_argument.append([i])

        indices = list(reversed(range(self.no_qubits)))

        return np.einsum(self.dm, indices, *trace_argument, optimize=True)

    def trace(self):
        tensor = np.array([1, 0, 0, 1])
        trace_argument = pytools.flatten(
            [[tensor, [i]] for i in range(self.no_qubits)])
        return np.einsum(self.dm, list(range(self.no_qubits)), *trace_argument, optimize=True)

    def project_measurement(self, bit, state):

        assert bit < self.no_qubits

        # the behaviour is a bit weird: swap the MSB to bit and then project out the highest one!
        projector = np.zeros(4)
        if state == 1:
            projector[3] = 1
        else:
            projector[0] = 1

        in_indices = list(reversed(range(self.no_qubits)))
        projector_indices = [bit]
        out_indices = list(reversed(range(self.no_qubits - 1)))
        if bit != self.no_qubits - 1:
            out_indices[-bit-1] = self.no_qubits - 1

        self.dm = np.einsum(self.dm, in_indices, projector, projector_indices, out_indices, optimize=True)

        self.no_qubits = len(self.dm.shape)

    def hadamard(self, bit):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.hadamard_ptm())

    def amp_ph_damping(self, bit, gamma, lamda):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.amp_ph_damping_ptm(gamma, lamda))

    def rotate_y(self, bit, angle):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.rotate_y_ptm(angle))

    def rotate_x(self, bit, angle):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.rotate_x_ptm(angle))

    def rotate_z(self, bit, angle):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.rotate_z_ptm(angle))

    def cphase(self, bit0, bit1):
        assert bit0 < self.no_qubits
        assert bit1 < self.no_qubits

        warnings.warn("deprecated, use two_ptm instead")
        two_ptm = ptm.double_kraus_to_ptm(np.diag([1, 1, 1, -1]))
        self.apply_two_ptm(bit0, bit1, two_ptm)
