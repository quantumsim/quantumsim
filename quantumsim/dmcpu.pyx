# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt

import numpy as np
from math import sqrt


cimport numpy as np


cdef class Density:

    cdef np.ndarray data_re
    cdef np.ndarray data_im
    cdef public int no_qubits
    cdef int size

    def __init__(self, no_qubits, data=None):
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
            self.data_re[0, 0] = 1
        else:
            raise ValueError("type of data not understood")
        
    def trace(self):
        return self.data_re.trace()

    def renormalize(self):
        trace = self.trace()
        self.data_re /= trace
        self.data_im /= trace

    def copy(self):
        dm = Density(self.no_qubits, self.to_array())
        return dm

    def to_array(self):
        return self.data_re + self.data_im*1j

    def get_diag(self):
        return self.data_re.diagonal()

    def cphase(self, bit0, bit1):
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

    def rotate_y(self, bit, cosine, sine):
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

                        na = cosine*a + sine*b
                        nb = -sine*a + cosine*b
                        nc = cosine*c + sine*d
                        nd = -sine*c + cosine*d

                        a = cosine*na + sine*nc
                        b = cosine*nb + sine*nd
                        c = -sine*na + cosine*nc
                        d = -sine*nb + cosine*nd

                        re[x, y] = a
                        re[x^mask, y] = b
                        re[x, y^mask] = c
                        re[x^mask, y^mask] = d

                        a = im[x, y]
                        b = im[x^mask, y]
                        c = im[x, y^mask]
                        d = im[x^mask, y^mask]

                        na = cosine*a + sine*b
                        nb = -sine*a + cosine*b
                        nc = cosine*c + sine*d
                        nd = -sine*c + cosine*d

                        a = cosine*na + sine*nc
                        b = cosine*nb + sine*nd
                        c = -sine*na + cosine*nc
                        d = -sine*nb + cosine*nd

                        im[x, y] = a
                        im[x^mask, y] = b
                        im[x, y^mask] = c
                        im[x^mask, y^mask] = d

    def add_ancilla(self, bit, anc_st):
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



