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

        if no_qubits > 15:
            raise ValueError("no_qubits=%d is way too many qubits, are you sure?"%no_qubits)

        if isinstance(data, np.ndarray):
            assert data.shape == (2**no_qubits, 2**no_qubits)
            data = data.astype(np.complex128)
            self.data = ga.to_gpu(data)
        elif isinstance(data, drv.DeviceAllocation):
            self.data = ga.empty((2**no_qubits, 2**no_qubits), dtype=np.complex128)
            drv.memcpy_dtod(self.data.gpudata, data, data.nbytes)
        elif isinstance(data, ga.GPUArray):
            assert data.shape == (2**no_qubits, 2**no_qubits)
            assert data.dtype == np.complex128
            self.data = data
        elif data is None:
            d = np.zeros((2**no_qubits, 2**no_qubits), dtype=np.complex128)
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




class Density10:
    def __init__(self, d9=None, a=0):
        """create a dm10 by adding ancilla in state a to a dm9
        dm9: a Density9 describing the state of 9 data qubits
        a: 1 or 0, the state of the ancilla added
        """
        pass
    
    def cphase(self, bit1, bit2):
        """Apply a cphase gate between bit1 and bit2
        bit1, bit2: integer between 0 and 9. 9 is the ancilla. "a" is a synonym for 9.
        """
        pass

    def hadamard(self, bit):
        """Apply a hadamard gate to bit.
        bit: integer between 0 and 9, or "a".
        """
        pass

    def amp_ph_damping(self, bit, params):
        """Apply a amplitude and phase damping channel to bit.
        bit: integer between 0 and 9, or "a".
        params: the damping probabilities (gamma, lambda)
        """
        pass


    def meas(self):
        """Measure the qubit. Return two unnormalized Density9 matrices with 
        traces corresponding to probabilities.
        """
        return (d9_0, d9_1)


class Density9:
    def __init__(self, data=None):
        """A density matrix describing the state of 9 data qubits
        data: a gpu array containing the dense density matrix. If None, 
              creata an initial density matrix with all qubits in ground state.
        """

    def trace(self):
        """Trace of the density matrix."""
        return tr
