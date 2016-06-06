import numpy as np


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
        self.data_re = self.data_re / self.trace()
        self.data_im = self.data_im / self.trace()

    def copy(self):
        dm = Density(self.no_qubits, self.to_array())

    def to_array(self):
        return self.data_re + self.data_im*1j

    def get_diag(self):
        return self.data_re.diagonal()

    def cphase(self, bit0, bit1):
        cdef unsigned int i, j

        cdef unsigned int mask0, mask1, mask_both

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        assert bit0 < self.no_qubits and bit1 < self.no_qubits

        re = self.data_re
        im = self.data_im

        mask = (1 << bit0) | (1 << bit1)

        for x in range(self.size):
            for y in range(x, self.size):
                if ((x & mask) == mask) != ((y & mask) == mask):
                    re[x, y] = -re[x, y]
                    im[x, y] = -im[x, y]




    def hadamard(self, bit):
        cdef unsigned int i, j, x, y
        cdef unsigned int lower_mask

        cdef np.ndarray[double, ndim=2] re
        cdef np.ndarray[double, ndim=2] im

        cdef double a, b, c, d
        cdef double na, nb, nc, nd

        lower_mask = (1 << bit) - 1
        mask = 1 << bit

        for i in range(self.size >> 1):
            for j in range(i, self.size >> 1):
                x = ((i & ~lower_mask) << 1) | (i & lower_mask)
                y = ((j & ~lower_mask) << 1) | (j & lower_mask)

                a = re[x, y]





    def amp_ph_damping(self, bit, gamma, lamda):
        pass

    def rotate_y(self, bit, cosine, sine):
        pass

    def add_ancilla(self, bit, anc_st):
        pass

    def measure_ancilla(self, bit):
        pass

