import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as ga


import pycuda.autoinit



# load the kernels
from pycuda.compiler import SourceModule
with open("./primitives.cu", "r") as f:
        mod = SourceModule(f.read())

_cphase = mod.get_function("cphase")
_hadamard = mod.get_function("hadamard")
_amp_ph_damping = mod.get_function("amp_ph_damping")
_dm_reduce = mod.get_function("dm_reduce")
_dm_inflate = mod.get_function("dm_inflate")
_get_diag = mod.get_function("get_diag")



class Density:
    def __init__(self, no_qubits, data=None):
        """create a new density matrix for several qubits. 
        no_qubits: number of qubits. 
        data: a numpy.ndarray, gpuarray.array, or pycuda.driver.DeviceAllocation. 
              must be of size (2**no_qubits, 2**no_qubits); is copied to GPU if not already there.
              Only upper triangle is relevant.
              If data is None, create a new density matrix with all qubits in ground state.
        """
        self.no_qubits = no_qubits
        self._block_size = 2**5
        self._grid_size = 2**max(no_qubits-5, 0)

        # self._size = max(self._block_size, 2**no_qubits)
        self._size = 2**no_qubits

        if no_qubits > 15:
            raise ValueError("no_qubits=%d is way too many qubits, are you sure?"%no_qubits)

        if isinstance(data, np.ndarray):
            assert data.shape == (self._size, self._size)
            data = data.astype(np.complex128)
            self.data = ga.to_gpu(data)
        elif isinstance(data, drv.DeviceAllocation):
            self.data = ga.empty((self._size, self._size), dtype=np.complex128)
            drv.memcpy_dtod(self.data.gpudata, data, data.nbytes)
        elif isinstance(data, ga.GPUArray):
            assert data.shape == (self._size, self._size)
            assert data.dtype == np.complex128
            self.data = data
        elif data is None:
            d = np.zeros((self._size, self._size), dtype=np.complex128)
            d[0,0] = 1
            self.data = ga.to_gpu(d)
        else:
            raise ValueError("type of data not understood")

    def trace(self):
        diag = ga.empty((2**self.no_qubits), dtype=np.float64)

        _get_diag(self.data.gpudata, diag.gpudata, np.uint32(self.no_qubits),
                block=(self._block_size,1,1), grid=(self._grid_size,1,1))

        trace = ga.sum(diag, dtype=np.float64).get()
        return trace

    def cphase(self, bit0, bit1):
        assert bit0 < self.no_qubits
        assert bit1 < self.no_qubits

        _cphase(self.data.gpudata,
                np.uint32((1<<bit0) | (1<<bit1)),
                np.uint32(self.no_qubits),
                block=(self._block_size,self._block_size,1),
                grid=(self._grid_size,self._grid_size,1))

    def hadamard(self, bit):
        assert bit < self.no_qubits
        
        _hadamard(self.data.gpudata, 
                np.uint32(1<<bit), 
                np.float64(0.5), 
                np.uint32(self.no_qubits),
                block=(self._block_size,self._block_size,1),
                grid=(self._grid_size,self._grid_size,1))

    def amp_ph_damping(self, bit, gamma, lamda):
        assert bit < self.no_qubits

        gamma = np.float64(gamma)
        lamda = np.float64(lamda)

        s1mgamma = np.float64(np.sqrt(1 - gamma))
        s1mlamda = np.float64(np.sqrt(1 - lamda))

        _amp_ph_damping(self.data.gpudata, 
                np.uint32(1<<bit), 
                gamma, s1mgamma, s1mlamda, 
                np.uint32(self.no_qubits),
                block=(self._block_size,self._block_size,1),
                grid=(self._grid_size,self._grid_size,1))

    def add_ancilla(self, bit, anc_st):
        assert bit <= self.no_qubits
        new_data = ga.zeros(
                (2**(self.no_qubits+1), 2**(self.no_qubits+1)),
                dtype=np.complex128)
        new_dm = Density(self.no_qubits + 1, new_data)

        dm9_0 = self.data
        dm9_1 = ga.zeros_like(self.data)
        if anc_st == 1:
            dm9_0, dm9_1 = dm9_1, dm9_0

        _dm_inflate(new_dm.data.gpudata, 
                np.uint32(bit), 
                dm9_0.gpudata, dm9_1.gpudata, 
                np.uint32(new_dm.no_qubits),
                block=(self._block_size, self._block_size,1),
                grid=(self._grid_size, self._grid_size,1))

        return new_dm

    def measure_ancilla(self, bit):
        assert bit < self.no_qubits

        d0 = ga.zeros((2**(self.no_qubits-1), 2**(self.no_qubits-1)), 
                np.complex128)
        d1 = ga.zeros((2**(self.no_qubits-1), 2**(self.no_qubits-1)), 
                np.complex128)

        _dm_reduce(self.data.gpudata, 
                np.uint32(bit),
                d0.gpudata, d1.gpudata,
                np.float64(1), np.float64(1), 
                np.uint32(self.no_qubits),
                block=(self._block_size, self._block_size,1),
                grid=(self._grid_size, self._grid_size,1))

        dm0 = Density(self.no_qubits - 1, d0)
        dm1 = Density(self.no_qubits - 1, d1)


        p0 = dm0.trace()
        p1 = dm1.trace()
        return p0, dm0, p1, dm1
