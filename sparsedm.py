from dm10 import Density

class SparseDM:
    def __init__(self, no_qubits):
        """A given set of qubit is kept in a state 
        """
        self.no_qubits = no_qubits
        self.classical = {bit: 0 for bit in range(no_qubits)}
        self.idx_in_full_dm = {}
        self.full_dm = Density(0)

        self.last_peak = None


    def ensure_dense(self, bit):
        """Make sure that the bit is removed from the classical bits and added to the
        density matrix, do nothing if it is already there."""
        if bit >= self.no_qubits:
            raise ValueError("This bit does not exist")
        if bit not in self.idx_in_full_dm:
            state = self.classical[bit]
            idx = self.full_dm.no_qubits
            self.full_dm = self.full_dm.add_ancilla(idx, state)
            del self.classical[bit]
            self.idx_in_full_dm[bit] = idx


    def peak_measurement(self, bit):
        """Calculate the two smaller density matrices that occur when 
        measuring qubit #bit. Return the probabilities. 

        The density matrices are stored and will be used by project_measurement
        if called with the same bit immediately afterwards
        """
        if bit in self.idx_in_full_dm:
            qbit = self.idx_in_full_dm[bit]
            p0, dm0, p1, dm1 = self.full_dm.measure_ancilla(qbit)
            self.last_peak = {'bit': bit, 0: dm0, 1: dm1}
            return (p0, p1)
        elif self.classical[bit] == 0:
            return (1, 0)
        elif self.classical[bit] == 1:
            return (0, 1)

    def project_measurement(self, bit, state): 
        """Project a bit to a fixed state, making it classical and 
        reducing the size of the full density matrix.
        The reduced density matrix is not normalized, so that
        its trace represents the probability for that event.
        """
        if bit in self.idx_in_full_dm:
            if self.last_peak == None or self.last_peak['bit'] != bit:
                self.peak_measurement(bit)
            self.full_dm = self.last_peak[state]
            self.classical[bit] = state
            for b in self.idx_in_full_dm:
                if self.idx_in_full_dm[b] > self.idx_in_full_dm[bit]:
                    self.idx_in_full_dm[b] -= 1
            del self.idx_in_full_dm[bit]
            self.last_peak = None
        else:
            raise ValueError("trying to measure classical bit")


    def cphase(self, bit0, bit1):
        """Apply a cphase gate between bit0 and bit1.
        """
        self.ensure_dense(bit0)
        self.ensure_dense(bit1)
        self.full_dm.cphase(self.idx_in_full_dm[bit0], 
                self.idx_in_full_dm[bit1])

    def hadamard(self, bit):
        """Apply a hadamard gate to qubit #bit.
        """
        self.ensure_dense(bit)
        self.full_dm.hadamard(self.idx_in_full_dm[bit])

    def amp_ph_damping(self, bit, gamma, lamda):
        """Apply amplitude and phase damping to qubit #bit.
        """
        self.ensure_dense(bit)
        self.full_dm.amp_ph_damping(self.idx_in_full_dm[bit], gamma, lamda)

    def trace(self):
        return self.full_dm.trace()
