# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt

import numpy as np
from math import sqrt


from . import ptm

cimport numpy as np


cdef class Density:
    """Density(no_qubits)

    Create a density matrix with the given number of qubits, labelled from 0 to no_qubits-1.
    """

    cdef public np.ndarray data
    cdef public int no_qubits
    cdef int size

    def __init__(self, no_qubits, data=None):
        """Density(no_qubits)

        Create a density matrix with the given number of qubits, labelled from 0 to no_qubits-1.

        This version runs on the CPU with operations implemented in cython.
        """
        self.no_qubits = no_qubits
        if no_qubits > 20:
            raise ValueError("Way too many qubits!")
        size = 2**no_qubits
        self.size = size

        if isinstance(data, np.ndarray):
            assert data.shape == (size, size)
            data = data.astype(np.complex128)

            self.data = np.zeros(size*size, dtype=np.float64)

            #convert to Pauli basis
            for i in range(self.no_qubits):
                self.to_pauli_basis(data, i)
            self.pauli_reshuffle(data, from_complex=True)

        elif data is None:
            self.data = np.zeros(size*size, np.float64)
            self.data[0] = 1
        else:
            raise ValueError("type of data not understood")

    def trace(self):
        """trace(self)

        Return the trace of the density matrix.
        """
        return self.data.reshape((self.size, self.size)).trace()
        
    def renormalize(self):
        """renormalize(self)

        Renormalize the density matrix to trace() == 1
        """
        trace = self.trace()
        self.data /= trace

    def copy(self):
        """copy(self)

        Return a copy of this density matrix.
        """
        dm = Density(self.no_qubits)
        dm.data = self.data.copy()
        return dm

    def to_array(self):
        """to_array(self)

        Return the data in the density matrix as a numpy array.
        """

        complex_dm = np.zeros((self.size, self.size), dtype=np.complex128)

        self.pauli_reshuffle(complex_dm, from_complex=False)
        for i in range(self.no_qubits):
            self.to_pauli_basis(complex_dm, i)

        return complex_dm

    def get_diag(self):
        """get_diag(self)
        
        Return the main diagonal of the density matrix as a numpy array.
        """
        return self.data.reshape((self.size, self.size)).diagonal()

    def apply_ptm(self, bit, ptm):
        """
        Apply a single qubit Pauli transfer matrix. 

        The PTM must be given in (0, x, y, 1) basis and as 4x4 reals.
        See also the quantumsim.ptm module.
        """
        cdef unsigned int il, jl, ih, jh, x, y

        cdef np.ndarray[np.float64_t, ndim=2] dt

        dt = self.data.reshape((self.size, self.size))

        assert bit < self.no_qubits

        cdef double a, b, c, d
        cdef double na, nb, nc, nd

        cdef unsigned int mask = (1<<bit)

        for ih in range(1<<(self.no_qubits - bit - 1)):
            for jh in range(1<<(self.no_qubits - bit - 1)):
                for il in range(1<<bit):
                    for jl in range(1<<bit):
                        x = (ih << (bit + 1)) | il
                        y = (jh << (bit + 1)) | jl

                        a = dt[x, y]
                        b = dt[x|mask, y]
                        c = dt[x, y|mask]
                        d = dt[x|mask, y|mask]


                        na, nb, nc, nd = np.dot(ptm, [a,b,c,d])

                        dt[x, y] = na
                        dt[x|mask, y] = nb
                        dt[x, y|mask] = nc
                        dt[x|mask, y|mask] = nd

    def cphase(self, bit0, bit1):
        """
        cphase(self, bit0, bit1)

        Apply a cphase gate acting on bit0 and bit1.

        bit0 and bit1 are zero-based indices identifying the qubits.

        (A cphase gate changes the phase of a wave function by pi if both 
        control bits are 1)
        """
        cdef unsigned int x, y

        cdef np.ndarray[np.float64_t, ndim=2] dt

        cdef double t, u

        cdef int idx0, idx1, mask

        assert bit0 < self.no_qubits and bit1 < self.no_qubits

        dt = self.data.reshape((self.size, self.size))

        mask = (1 << bit0) | (1 << bit1)

        for x in range(self.size):
            for y in range(self.size):
                    idx0 = 2*(( x >> bit0) & 1) | ((y >> bit0) & 1)
                    idx1 = 2*(( x >> bit1) & 1) | ((y >> bit1) & 1)


                    if idx0 == 3 and idx1 != 0:
                        dt[x, y] = - dt[x, y]
                    if idx1 == 3 and idx0 != 0:
                        dt[x, y] = - dt[x, y]

                    if idx1 == 1 and (idx0 == 1 or idx0 == 2):
                        t = -dt[x, y]
                        u = -dt[x^mask, y^mask]
                        dt[x^mask, y^mask] = t
                        dt[x, y] = u



    def hadamard(self, bit):
        """hadamard(self, bit)

        Apply a Hadamard gate to bit (0..no_qubits-1).

        (A Hadamard gate is defined by the matrix 
        
            [1, 1; 1 -1]/sqrt(2) 
        
        in computational basis.)
        """
        self.apply_ptm(bit, ptm.hadamard_ptm())

    def amp_ph_damping(self, bit, gamma, lamda):
        """amp_ph_damping(self, bit, gamma, lamda)

        Apply amplitude and phase damping to bit (0..no_qubits).

        Amplitude and phase damping is characterized by the constants gamma and lambda.

        We have gamma = 1 - p1 = 1 - exp(-t/T1)
                lamda = 1 - p2 = 1 - exp(-t/T2)


        where t is the duration of decay and T1, T2 are properties of the qubit.
        
        (the stupid misspelling of lamda is because "lambda" is a keyword in python)
        """
        self.apply_ptm(bit, ptm.amp_ph_damping_ptm(gamma, lamda))

    def rotate_x(self, bit, angle):
        """rotate_x(self, bit, angle)

        Rotation on the bloch sphere around the x-axis by angle theta.

        bit: which bit is affected (0..no_qubits)
        angle: the angle of rotation
        """
        self.apply_ptm(bit, ptm.rotate_x_ptm(angle))


    def rotate_y(self, bit, angle):
        """rotate_y(self, bit, angle)

        Rotation on the bloch sphere around the y-axis by angle theta.

        bit: which bit is affected (0..no_qubits)
        angle: the angle of rotation
        """
        self.apply_ptm(bit, ptm.rotate_y_ptm(angle))

    def rotate_z(self, bit, angle):
        """rotate_z(self, bit, angle)

        Rotation on the bloch sphere around the z-axis by angle theta.

        bit: which bit is affected (0..no_qubits)
        angle: the angle of rotation
        """
        self.apply_ptm(bit, ptm.rotate_z_ptm(angle))

    def add_ancilla(self, bit, anc_st):
        """add_ancilla(self, bit, anc_st)

        Returns a new Density with an ancilla added to the density matrix, increasing no_qubits by 1.

        bit: The index of the new ancilla (0..no_qubits)
        anc_st: The state of the newly added ancilla (0 or 1)
        """
        cdef unsigned int i, j

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

        cdef np.ndarray[double, ndim=2] re_new
        cdef np.ndarray[double, ndim=2] im_new

        assert bit < self.no_qubits + 1

        new_size = self.size << 1

        re_new = np.zeros((new_size, new_size), np.float64)
        im_new = np.zeros((new_size, new_size), np.float64)

        cdef unsigned int lower_mask, upper_mask, bit_mask
        cdef unsigned int new_x, new_y

        lower_mask = (1 << bit) - 1
        upper_mask = ~lower_mask
        bit_mask = anc_st << bit

        for i in range(self.size):
            for j in range(self.size):
                new_x = ((i & upper_mask) << 1) | (i & lower_mask) | bit_mask
                new_y = ((j & upper_mask) << 1) | (j & lower_mask) | bit_mask

                re_new[new_x, new_y] = re[i, j]
                im_new[new_x, new_y] = im[i, j]

        return Density(self.no_qubits + 1, data=re_new + 1j*im_new)

    def measure_ancilla(self, bit):
        """measure_ancilla(self, bit)

        Calculate the projections when measuring bit (0..no_qubits-1).

        Returns (p0, r0, p1, r1), where pi is the probability to measure outcome i,
        and ri is a the reduced Density of the other bits, given that outcome.

        (We have pi = ri.trace()) 
        """
        assert bit < self.no_qubits

        new_size = self.size >> 1

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

        cdef np.ndarray[double, ndim=2] re_new0, re_new1
        cdef np.ndarray[double, ndim=2] im_new0, im_new1


        re_new0 = np.zeros((new_size, new_size), np.float64)
        re_new1 = np.zeros((new_size, new_size), np.float64)
        im_new0 = np.zeros((new_size, new_size), np.float64)
        im_new1 = np.zeros((new_size, new_size), np.float64)

        cdef unsigned int i, j

        cdef unsigned int lower_mask, upper_mask, bit_mask
        cdef unsigned int from_x, from_y, to_x, to_y

        lower_mask = (1 << bit) - 1
        upper_mask = ~lower_mask
        bit_mask = 1 << bit

        for i in range(new_size):
            for j in range(new_size):
                from_x = ((i & upper_mask) << 1) | (i&lower_mask) 
                from_y = ((j & upper_mask) << 1) | (j&lower_mask) 

                re_new0[i, j] = re[from_x, from_y]
                im_new0[i, j] = im[from_x, from_y]

                from_x |= bit_mask
                from_y |= bit_mask

                re_new1[i, j] = re[from_x, from_y]
                im_new1[i, j] = im[from_x, from_y]

        dm0 = Density(self.no_qubits - 1, re_new0 + im_new0*1j)
        dm1 = Density(self.no_qubits - 1, re_new1 + im_new1*1j)

        tr0 = dm0.trace()
        tr1 = dm1.trace()

        return tr0, dm0, tr1, dm1

    def to_pauli_basis(self, complex_dm, bit):
        cdef unsigned int il, jl, ih, jh, x, y

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = complex_dm.real
        im = complex_dm.imag

        assert bit < self.no_qubits

        cdef double a, b, c, d
        cdef double na, nb, nc, nd

        cdef unsigned int mask = (1<<bit)

        for ih in range(1<<(self.no_qubits - bit - 1)):
            for jh in range(1<<(self.no_qubits - bit - 1)):
                for il in range(1<<bit):
                    for jl in range(1<<bit):
                        x = (ih << (bit + 1)) | il
                        y = (jh << (bit + 1)) | jl


                        a = re[x, y]
                        b = re[x|mask, y]
                        c = re[x, y|mask]
                        d = re[x|mask, y|mask]

                        na = a
                        nb = (b+c)/np.sqrt(2)
                        nc = (b-c)/np.sqrt(2)
                        nd = d

                        re[x, y] = na
                        re[x|mask, y] = nb
                        re[x, y|mask] = nc
                        re[x|mask, y|mask] = nd

                        a = im[x, y]
                        b = im[x|mask, y]
                        c = im[x, y|mask]
                        d = im[x|mask, y|mask]

                        na = a
                        nb = (b+c)/np.sqrt(2)
                        nc = (b-c)/np.sqrt(2)
                        nd = d

                        im[x, y] = na
                        im[x|mask, y] = nb
                        im[x, y|mask] = nc
                        im[x|mask, y|mask] = nd
    
    def pauli_reshuffle(self, complex_dm, from_complex=True):
        cdef int i
        cdef int v
        cdef int x, y, addr, addr_new

        cdef float f

        cdef np.ndarray[np.float64_t] re
        cdef np.ndarray[np.float64_t] im


        complex_dm = complex_dm.ravel()

        re = complex_dm.real
        im = complex_dm.imag

        for x in range(1<<self.no_qubits):
            for y in range(1<<self.no_qubits):
                v = ~x & y
                v ^= v >> 1
                v ^= v >> 2
                v = (v & 0x11111111U) * 0x11111111U
                v = (v >> 28) & 1

                addr = (x << self.no_qubits) | y
                # addr_new = 0
                # for i in range(16):
                    # addr_new |= (x & 1 << i) << i | (y & 1 << i) << (i + 1)
                addr_new = addr

                if from_complex:
                    if v == 1:
                        self.data[addr_new] = im[addr]
                    else:
                        self.data[addr_new] = re[addr]
                else:
                    if v == 1:
                        complex_dm.imag[addr] = self.data[addr_new]
                    else:
                        complex_dm.real[addr] = self.data[addr_new]

            
                        




                

