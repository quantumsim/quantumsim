# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt

import numpy as np

from collections import defaultdict

try:
    from . import dm10
    using_gpu = True
except ImportError:
    from . import dmcpu as dm10
    using_gpu = False

class SparseDM:
    def __init__(self, names=None):
        """A sparse density matrix for a set of qubits with names `names`. 

        Each qubit can be in a "classical state", where it is in a basis state 
        (i.e. 0 or 1) and not correlated with other qubits, so that the total density matrix 
        can be written as a product state. This is the case after a measurement projection,
        meaning that a measurement turns a qubit classical.

        If a qubit is not classical, it is quantum, which means that it is part of the 
        full dense density matrix `self.full_dm`.
        """
        if isinstance(names, int):
            names = list(range(names))

        self.names = names
        self.no_qubits = len(names)
        self.classical = {bit: 0 for bit in names}
        self.idx_in_full_dm = {}
        self.full_dm = dm10.Density(0)
        self.max_bits_in_full_dm = 0

        self.classical_probability = 1

        self.single_ptms_to_do = defaultdict(list)

        self.last_peak = None
        self._last_majority_vote_array = None
        self._last_majority_vote_mask = None

    def ensure_dense(self, bit):
        """Make sure that the bit is removed from the classical bits and added to the
        density matrix, do nothing if it is already there. Does not change the state of the system.
        """
        if bit not in self.names:
            raise ValueError("This bit does not exist")
        if bit not in self.idx_in_full_dm:
            state = self.classical[bit]
            idx = self.full_dm.no_qubits
            self.full_dm.add_ancilla(state)
            del self.classical[bit]
            self.idx_in_full_dm[bit] = idx

            new_max = max(self.max_bits_in_full_dm, len(self.idx_in_full_dm))
            self.max_bits_in_full_dm = new_max

    def ensure_classical(self, bit, epsilon=1e-7):
        """Try to make a qubit classical. This only succeeds if the qubit already is classical, or if a measurement
        returns a definite result with fidelity > (1-epsilon), in which case the measurement is performed with the probable outcome.
        """
        self.combine_and_apply_single_ptm(bit)
        if bit not in self.names:
            raise ValueError("This bit does not exist")
        if bit in self.idx_in_full_dm:
            p0, p1 = self.peak_measurement(bit)
            if p0 < epsilon:
                self.project_measurement(bit, 1)
            elif p1 < epsilon:
                self.project_measurement(bit, 0)
            else:
                raise ValueError("Trying to classicalize entangled quantum bit")

    def peak_measurement(self, bit):
        """Calculate the two smaller density matrices that occur when 
        measuring qubit #bit. Return the probabilities (not normalized to one).

        The density matrices are stored internally and will be used 
        without recalculation if `project_measurement` is called with the same bit immediately afterwards.
        """
        self.combine_and_apply_single_ptm(bit)
        if bit in self.idx_in_full_dm:
            qbit = self.idx_in_full_dm[bit]
            p0, p1 = self.full_dm.partial_trace(qbit)
            return (p0, p1)
        elif self.classical[bit] == 0:
            return (1, 0)
        elif self.classical[bit] == 1:
            return (0, 1)

    def project_measurement(self, bit, state): 
        """Project a bit to a fixed state, making it classical and 
        reducing the size of the full density matrix.
        The reduced density matrix is not normalized, so that
        its trace after projection represents the probability for that event.
        """
        self.combine_and_apply_single_ptm(bit)
        if bit in self.idx_in_full_dm:
            self.full_dm.project_measurement(self.idx_in_full_dm[bit], state)
            self.classical[bit] = state
            for b in self.idx_in_full_dm:
                if self.idx_in_full_dm[b] == self.full_dm.no_qubits:
                    self.idx_in_full_dm[b] = self.idx_in_full_dm[bit] 
            del self.idx_in_full_dm[bit]
        else:
            raise ValueError("trying to measure classical bit")
    
    def peak_multiple_measurements(self, bits):
        """Obtain the probabilities for all combinations of a multiple
        qubit measurement. Act on a copy, do not destroy this density matrix.

        bits is a list of qubit names. Return a list with up to `2**len(bits)` tuples of the form

        [(result, probability), ...] 

        where `result` is a dict describing the measurement result {"bit0": 1, "bit2": 0, ...}, 
        and `probability` is the corresponding probability. 

        If results are omitted from this list, the corresponding probability is assumed to be 0.

        Note that these probabilities are not normalized if previous projections took place.
        """

        for bit in bits: 
            self.combine_and_apply_single_ptm(bit)

        classical_bits = {bit: self.classical[bit] for bit in bits if bit in self.classical}

        res = [(classical_bits, self.full_dm.copy())]

        bits = [bit for bit in bits if bit not in self.classical]

        bit_idxs = [(bit, self.idx_in_full_dm[bit]) for i,bit in enumerate(bits)]

        mask = 0
        for bit in bits:
            mask |= 1 << self.idx_in_full_dm[bit]

        diagonal = self.full_dm.get_diag()

        probs = {}

        for idx, prob in enumerate(diagonal):
            if idx & mask in probs:
                probs[idx & mask]  += prob
            else: 
                probs[idx & mask]  = prob

        res = []
        for idx in probs:
            outcome = classical_bits.copy()
            for bit in bits: 
                outcome[bit] = int(idx & (1 << self.idx_in_full_dm[bit])>0)

            res.append((outcome, probs[idx] * self.classical_probability))

        return res

    def trace(self):
        """Return the trace of the density matrix, which is the probability for all measurement projections in the history.
        """
        return self.classical_probability * self.full_dm.trace()

    def renormalize(self):
        """Renormalize the density matrix to trace 1.
        """
        self.full_dm.renormalize()
        self.classical_probability = 1

    def copy(self):
        """Return an identical but distinct copy of this object.

        If a measurement has been peaked at, the reduced density matrices are discarded.
        """

        cp = SparseDM(self.names)
        cp.single_ptms_to_do = self.single_ptms_to_do
        cp.classical = self.classical.copy()
        cp.idx_in_full_dm = self.idx_in_full_dm.copy()
        cp.last_peak = None
        cp.full_dm = self.full_dm.copy()

        return cp

    def cphase(self, bit0, bit1):
        """Apply a cphase gate between bit0 and bit1.
        """
        self.ensure_dense(bit0)
        self.ensure_dense(bit1)
        self.combine_and_apply_single_ptm(bit0)
        self.combine_and_apply_single_ptm(bit1)
        self.full_dm.cphase(self.idx_in_full_dm[bit0], 
                self.idx_in_full_dm[bit1])

    def apply_all_pending(self):
        for bit in list(self.single_ptms_to_do.keys()):
            self.combine_and_apply_single_ptm(bit)


    def combine_and_apply_single_ptm(self, bit):
        if bit in self.single_ptms_to_do:
            self.ensure_dense(bit)
            ptm = self.single_ptms_to_do[bit][0]
            for ptm2 in self.single_ptms_to_do[bit][1:]:
                ptm = ptm2.dot(ptm)
            self.full_dm.apply_ptm(self.idx_in_full_dm[bit], ptm)

            del self.single_ptms_to_do[bit]

    def apply_ptm(self, bit, ptm):
        self.single_ptms_to_do[bit].append(ptm)

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

    def rotate_x(self, bit, angle):
        """Apply a rotation around the x-axis of the Bloch sphere of bit `bit` 
        by `angle` (in radians).
        """
        self.ensure_dense(bit)
        self.full_dm.rotate_x(self.idx_in_full_dm[bit], angle)

    def rotate_y(self, bit, angle):
        """Apply a rotation around the y-axis of the Bloch sphere of bit `bit` 
        by `angle` (in radians).
        """
        self.ensure_dense(bit)
        self.full_dm.rotate_y(self.idx_in_full_dm[bit], angle)

    def rotate_z(self, bit, angle):
        """Apply a rotation around the z-axis of the Bloch sphere of bit `bit` 
        by `angle` (in radians).
        """
        self.ensure_dense(bit)
        self.full_dm.rotate_z(self.idx_in_full_dm[bit], angle)

    def set_bit(self, bit, value):
        """Set the value of a classical bit to `value` (0 or 1).
        """
        self.ensure_classical(bit)
        self.classical[bit] = value

    def majority_vote(self, bits):
        """Return the (total) probability for measuring more than half 1 (or a specified result)
        when measuring all the given bits. Do not actually perform the measurement.

        This is evaluated as Tr(ρ.M) with M a diagonal matrix. 
        The diagonal of M is constructed when needed, then cached and reused when possible.

        bits: any iterable containing bit names, then return the probability for a majority of one measurements.
              if bits is a dict of the form {bit1: result1, bit2: result2 , ...} with result in {0, 1}, 
              return the probabilty for the majority of the measurements coinciding with the given result.
        """

        dense_bits = {b for b in bits if b in self.idx_in_full_dm}

        for bit in bits:
            self.combine_and_apply_single_ptm(bit)

        bit_result = {}
        for b in bits:
            try:
                v = bits.get(b)
                if v not in [0, 1]:
                    raise ValueError("Measurement result can only be 0 or 1, not "+str(v))
                bit_result[b] = v
            except AttributeError:
                bit_result[b] = 1

        classical_bits_sum = sum(self.classical[b] ^ (1-bit_result[b])
                for b in bits if b in self.classical)

        mask = 0
        result_mask = 0
        for b in dense_bits:
            mask += 1 << self.idx_in_full_dm[b]
            result_mask |= ((1-bit_result[b]) << self.idx_in_full_dm[b])

        if 1: #mask != self._last_majority_vote_mask:
            adresses = np.arange(2**len(self.idx_in_full_dm)) ^ result_mask
            majority = np.zeros_like(adresses)
            for _ in range(len(self.idx_in_full_dm)):
                majority += adresses & 1
                adresses >>= 1

            self._last_majority_vote_mask = mask
            self._last_majority_vote_array = majority
        else:
            majority = self._last_majority_vote_array

        majority = (majority+classical_bits_sum > len(bits)/2).astype(np.int)

        diag = self.full_dm.get_diag()

        return np.dot(majority, diag)*self.classical_probability
