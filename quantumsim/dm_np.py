import numpy as np
import pytools


class DensityNP:

    def __init__(self, no_qubits, data=None):
        self.no_qubits = no_qubits
        self.shape = [4] * no_qubits

        if data:
            raise NotImplementedError
        else:
            self.dm = np.zeros(self.shape)
            self.dm[tuple([0] * self.no_qubits)] = 1

    def renormalize(self):
        self.dm = self.dm / self.trace()

    def copy(self):
        cp = DensityNP(no_qubits=self.no_qubits)
        cp.dm = self.dm.copy()
        return cp

    def to_array(self):
        pass

    def get_diag(self):
        pass

    def apply_two_ptm(self, bit0, bit1, two_ptm):
        two_ptm = two_ptm.reshape((4, 4, 4, 4))
        dummy_idx0, dummy_idx1 = self.no_qubits, self.no_qubits + 1
        out_indices = list(reversed(range(self.no_qubits)))
        in_indices = list(reversed(range(self.no_qubits)))
        in_indices[self.no_qubits - bit0 - 1] = dummy_idx0
        in_indices[self.no_qubits - bit1 - 1] = dummy_idx1
        two_ptm_indices = [
            dummy_idx0, dummy_idx1,
            self.no_qubits - bit0 - 1,
            self.no_qubits - bit1 - 1
        ]
        self.dm = np.einsum(
            self.dm, in_indices, two_ptm, two_ptm_indices, out_indices)

    def apply_ptm(self, bit, one_ptm):
        dummy_idx = self.no_qubits
        out_indices = list(range(self.no_qubits))
        in_indices = list(range(self.no_qubits))
        in_indices[self.no_qubits - bit - 1] = dummy_idx
        ptm_indices = [self.no_qubits - bit - 1, dummy_idx]
        self.dm = np.einsum(self.dm, in_indices, one_ptm, ptm_indices, out_indices)

    def add_ancilla(self, anc_st):
        anc_dm = np.zeros(4)
        if anc_st == 1:
            anc_dm[3] = 1
        else:
            anc_dm[0] = 1
        self.dm = np.einsum(
            anc_dm, [0], self.dm, list(range(1, self.no_qubits + 1)))
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

        return np.einsum(self.dm, indices, *trace_argument)

    def trace(self):
        tensor = np.array([1, 0, 0, 1])
        trace_argument = pytools.flatten(
            [[tensor, [i]] for i in range(self.no_qubits)])
        return np.einsum(self.dm, list(range(self.no_qubits)), *trace_argument)

    def project_measurement(self, bit, state):
        projector = np.zeros(4)
        if state == 1:
            projector[3] = 1
        else:
            projector[0] = 1

        dummy_idx = self.no_qubits
        out_indices = list(reversed(range(self.no_qubits)))
        in_indices = list(reversed(range(self.no_qubits)))
        in_indices[self.no_qubits - bit - 1] = dummy_idx
        projector_indices = [dummy_idx]
        self.dm = np.einsum(self.dm, in_indices, projector, projector_indices, out_indices)
