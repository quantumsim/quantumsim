import numpy as np


cdef class Density:

    cdef np.ndarray data
    cdef int no_qubits

    def __init__(self, no_qubits, data=None):
        self.no_qubits = no_qubits
        size = 2**no_qubits

        if isinstance(data, np.ndarray):
            assert data.shape == (size, size)
            data = data.astype(np.complex128)
            self.data = data
        elif data is None:
            self.data = np.zeros((size, size), np.complex128)
            self.data[0, 0] = 1
        else:
            raise ValueError("type of data not understood")
        

    def trace(self):
        return self.data.trace()

    def renormalize(self):
        self.data = self.data / self.trace()

    def copy(self):
        dm = Density(self.no_qubits, self.data.copy())

    def to_array(self):
        return self.data

    def get_diag(self):
        return self.data.diagonal()
    

    def cphase(self, bit0, bit1):
        pass

    def hadamard(self, bit):
        pass

    def amp_ph_damping(self, bit, gamma, lamda):
        pass

    def rotate_y(self, bit, cosine, sine):
        pass

    def add_ancilla(self, bit, anc_st):
        pass

    def measure_ancilla(self, bit):
        pass

