# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt

import numpy as np
from math import sqrt


cimport numpy as np


cdef class Density:
    """Density(no_qubits)

    Create a density matrix with the given number of qubits, labelled from 0 to no_qubits-1.
    """

    cdef np.ndarray data_re
    cdef np.ndarray data_im
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
            self.data_re = data.real
            self.data_im = data.imag
        elif data is None:
            self.data_re = np.zeros((size, size), np.float64)
            self.data_im = np.zeros((size, size), np.float64)
            assert np.allclose(self.data_im, 0)
            assert np.allclose(self.data_re, 0)
            self.data_re[0, 0] = 1
        else:
            raise ValueError("type of data not understood")
        
    def trace(self):
        """trace(self)

        Return the trace of the density matrix.
        """
        return self.data_re.trace()

    def renormalize(self):
        """renormalize(self)

        Renormalize the density matrix to trace() == 1
        """
        trace = self.trace()
        self.data_re /= trace
        self.data_im /= trace

    def copy(self):
        """copy(self)

        Return a copy of this density matrix.
        """
        dm = Density(self.no_qubits, self.to_array())
        return dm

    def to_array(self):
        """to_array(self)

        Return the data in the density matrix as a numpy array.
        """
        return self.data_re + self.data_im*1j

    def get_diag(self):
        """get_diag(self)
        
        Return the main diagonal of the density matrix as a numpy array.
        """
        return self.data_re.diagonal()

    def cphase(self, bit0, bit1):
        """
        cphase(self, bit0, bit1)

        Apply a cphase gate acting on bit0 and bit1.

        bit0 and bit1 are zero-based indices identifying the qubits.

        (A cphase gate changes the phase of a wave function by pi if both 
        control bits are 1)
        """
        cdef unsigned int x, y

        cdef unsigned int mask0, mask1, mask_both

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

        assert bit0 < self.no_qubits and bit1 < self.no_qubits

        mask = (1 << bit0) | (1 << bit1)

        for x in range(self.size):
            for y in range(self.size):
                if ((x & mask) == mask) != ((y & mask) == mask):
                    re[x, y] = -re[x, y]
                    im[x, y] = -im[x, y]

    def hadamard(self, bit):
        """hadamard(self, bit)

        Apply a Hadamard gate to bit (0..no_qubits-1).

        (A Hadamard gate is defined by the matrix [1, 1; 1 -1]/sqrt(2) in computational basis.)
        """
        cdef unsigned int il, jl, ih, jh, x, y

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

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

                        na = a+b+c+d
                        nb = a-b+c-d
                        nc = a+b-c-d
                        nd = a-b-c+d

                        re[x, y] = 0.5*na
                        re[x|mask, y] = 0.5*nb
                        re[x, y|mask] = 0.5*nc
                        re[x|mask, y|mask] = 0.5*nd

                        a = im[x, y]
                        b = im[x|mask, y]
                        c = im[x, y|mask]
                        d = im[x|mask, y|mask]

                        na = a+b+c+d
                        nb = a-b+c-d
                        nc = a+b-c-d
                        nd = a-b-c+d

                        im[x, y] = 0.5*na
                        im[x|mask, y] = 0.5*nb
                        im[x, y|mask] = 0.5*nc
                        im[x|mask, y|mask] = 0.5*nd

    def amp_ph_damping(self, bit, gamma, lamda):
        """amp_ph_damping(self, bit, gamma, lamda)

        Apply amplitude and phase damping to bit (0..no_qubits).

        Amplitude and phase damping is characterized by the constants gamma and lambda.

        We have gamma = 1 - p1 = 1 - exp(-t/T1)
                lamda = 1 - p2 = 1 - exp(-t/T2)


        where t is the duration of decay and T1, T2 are properties of the qubit.
        
        (the stupid misspelling of lamda is because "lambda" is a keyword in python)
        """
        cdef unsigned int ih, jh, il, jl

        cdef double dgamma, dlamda, ds1mgamma, ds1mlamda
        dgamma = gamma
        dlamda = lamda
        ds1mgamma = sqrt(1 - gamma)
        ds1mlamda = sqrt(1 - lamda)

        assert np.allclose(dgamma + ds1mgamma**2, 1)
        assert np.allclose(dlamda + ds1mlamda**2, 1)

        assert bit < self.no_qubits

        cdef double a, b, c, d
        cdef double na, nb, nc, nd

        cdef unsigned int mask = (1<<bit)

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

        for ih in range(1<<(self.no_qubits - bit - 1)):
            for jh in range(1<<(self.no_qubits - bit - 1)):
                for il in range(1<<bit):
                    for jl in range(1<<bit):
                        x = (ih << (bit + 1)) | il
                        y = (jh << (bit + 1)) | jl

                        a = re[x, y]
                        b = re[x^mask, y]
                        c = re[x, y^mask]
                        d = re[x^mask, y^mask]

                        na = a + dgamma*d
                        nb = b*ds1mgamma*ds1mlamda
                        nc = c*ds1mgamma*ds1mlamda
                        nd = d - dgamma*d

                        re[x, y] = na
                        re[x^mask, y] = nb
                        re[x, y^mask] = nc
                        re[x^mask, y^mask] = nd

                        a = im[x, y]
                        b = im[x^mask, y]
                        c = im[x, y^mask]
                        d = im[x^mask, y^mask]

                        na = a + dgamma*d
                        nb = b*ds1mgamma*ds1mlamda
                        nc = c*ds1mgamma*ds1mlamda
                        nd = d - dgamma*d

                        im[x, y] = na
                        im[x^mask, y] = nb
                        im[x, y^mask] = nc
                        im[x^mask, y^mask] = nd

    def rotate_x(self, bit, cos, sin):
        """rotate_x(self, bit, cos, sin)

        Rotation on the bloch sphere around the x-axis by angle theta.

        bit: which bit is affected (0..no_qubits)
        cos: The cosine of theta/2
        sin: The sine of theta/2
        """
        cdef unsigned int il, jl, ih, jh, x, y

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

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
                        b = im[x^mask, y]
                        c = im[x, y^mask]
                        d = re[x^mask, y^mask]

                        na = cos*cos*a + sin*cos*(b-c) + sin*sin*d
                        nb = -sin*cos*(a-d) + sin*sin*c + cos*cos*b
                        nc = sin*cos*(a-d) + sin*sin*b + cos*cos*c
                        nd = cos*cos*d + sin*cos*(c-b) + sin*sin*a

                        re[x, y] = na
                        im[x^mask, y] = nb
                        im[x, y^mask] = nc 
                        re[x^mask, y^mask] = nd

                        a = im[x, y]
                        b = re[x^mask, y]
                        c = re[x, y^mask]
                        d = im[x^mask, y^mask]

                        na = cos*cos*a - sin*cos*(b-c) + sin*sin*d
                        nb = -sin*cos*(d-a) + sin*sin*c + cos*cos*b
                        nc = sin*cos*(d-a) + sin*sin*b + cos*cos*c
                        nd = cos*cos*d - sin*cos*(c-b) + sin*sin*a

                        im[x, y] = na
                        re[x^mask, y] = nb
                        re[x, y^mask] = nc 
                        im[x^mask, y^mask] = nd

    def rotate_y(self, bit, cos, sin):
        """rotate_y(self, bit, cos, sin)

        Rotation on the bloch sphere around the y-axis by angle theta.

        bit: which bit is affected (0..no_qubits)
        cos: The cosine of theta/2
        sin: The sine of theta/2
        """
        cdef unsigned int il, jl, ih, jh, x, y

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

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
                        b = re[x^mask, y]
                        c = re[x, y^mask]
                        d = re[x^mask, y^mask]

                        na = cos*cos*a + cos*sin*(b+c) + sin*sin*d
                        nb = sin*cos*(d-a) + cos*cos*b - sin*sin*c
                        nc = sin*cos*(d-a) + cos*cos*c - sin*sin*b
                        nd = sin*sin*a - cos*sin*(b+c) + cos*cos*d

                        re[x, y] = na
                        re[x^mask, y] = nb
                        re[x, y^mask] = nc
                        re[x^mask, y^mask] = nd

                        a = im[x, y]
                        b = im[x^mask, y]
                        c = im[x, y^mask]
                        d = im[x^mask, y^mask]

                        na = cos*cos*a + cos*sin*(b+c) + sin*sin*d
                        nb = sin*cos*(d-a) + cos*cos*b - sin*sin*c
                        nc = sin*cos*(d-a) + cos*cos*c - sin*sin*b
                        nd = sin*sin*a - cos*sin*(b+c) + cos*cos*d

                        im[x, y] = na
                        im[x^mask, y] = nb
                        im[x, y^mask] = nc
                        im[x^mask, y^mask] = nd

    def rotate_z(self, bit, cos2, sin2):
        """rotate_z(self, bit, cos2, sin2)

        Rotation on the bloch sphere around the z-axis by angle theta.

        bit: which bit is affected (0..no_qubits)
        cos: The cosine of theta
        sin: The sine of theta
        """
        cdef unsigned int il, jl, ih, jh, x, y

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        re = self.data_re
        im = self.data_im

        assert bit < self.no_qubits

        cdef double b_re, b_im, c_re, c_im
        cdef double nb_re, nb_im, nc_re, nc_im

        cdef unsigned int mask = (1<<bit)

        for ih in range(1<<(self.no_qubits - bit - 1)):
            for jh in range(1<<(self.no_qubits - bit - 1)):
                for il in range(1<<bit):
                    for jl in range(1<<bit):
                        x = (ih << (bit + 1)) | il
                        y = (jh << (bit + 1)) | jl


                        b_re = re[x^mask, y]
                        b_im = im[x^mask, y]
                        c_re = re[x, y^mask]
                        c_im = im[x, y^mask]

                        nb_re = cos2*b_re - sin2*b_im
                        nb_im = cos2*b_im + sin2*b_re
                        nc_re = cos2*c_re + sin2*c_im
                        nc_im = cos2*c_im - sin2*c_re

                        re[x^mask, y] = nb_re
                        im[x^mask, y] = nb_im
                        re[x, y^mask] = nc_re
                        im[x, y^mask] = nc_im

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



