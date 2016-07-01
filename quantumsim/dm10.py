# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or https://www.gnu.org/licenses/gpl.txt

import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as ga


import pycuda.autoinit


# load the kernels
from pycuda.compiler import SourceModule

import sys
import os 

package_path = os.path.dirname(os.path.realpath(__file__))

mod = None
for kernel_file in [sys.prefix+"/pycudakernels/primitives.cu", package_path + "/primitives.cu"]:
    try:
        with open(kernel_file, "r") as kernel_source_file:
            mod = SourceModule(kernel_source_file.read(), options=["--default-stream", "per-thread"])
            break
    except FileNotFoundError:
        pass

if mod is None:
    raise ImportError("could not find primitives.cu")


_cphase = mod.get_function("cphase")
_cphase.prepare("PII")
_hadamard = mod.get_function("hadamard")
_hadamard.prepare("PIdI")
_amp_ph_damping = mod.get_function("amp_ph_damping")
_amp_ph_damping.prepare("PIdddI")
_dm_reduce = mod.get_function("dm_reduce")
_dm_reduce.prepare("PIPPddI")
_dm_inflate = mod.get_function("dm_inflate")
_dm_inflate.prepare("PIPPI")
_get_diag = mod.get_function("get_diag")
_get_diag.prepare("PPI")
_rotate_y = mod.get_function("rotate_y")
_rotate_y.prepare("PIddI")
_rotate_x = mod.get_function("rotate_x")
_rotate_x.prepare("PIddI")
_rotate_z = mod.get_function("rotate_z")
_rotate_z.prepare("PIddI")


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
        self._block_size = 2**3
        self._grid_size = 2**max(no_qubits - 3, 0)

        # self._size = max(self._block_size, 2**no_qubits)
        self._size = 2**no_qubits

        if no_qubits > 15:
            raise ValueError(
                "no_qubits=%d is way too many qubits, are you sure?" % no_qubits)

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
            d[0, 0] = 1
            self.data = ga.to_gpu(d)
        else:
            raise ValueError("type of data not understood")

    def trace(self):
        diag = ga.empty((2**self.no_qubits), dtype=np.float64)
        block = (self._block_size, 1, 1)
        grid = (self._grid_size, 1, 1)

        _get_diag.prepared_call(grid, block,
                                self.data.gpudata, diag.gpudata, np.uint32(self.no_qubits))

        trace = ga.sum(diag, dtype=np.float64).get()
        return trace

    def renormalize(self):
        """Renormalize to trace one."""
        tr = self.trace()
        self.data *= np.float(1 / tr)

    def copy(self):
        "Return a deep copy of this Density."

        data_cp = self.data.copy()
        cp = Density(self.no_qubits, data=data_cp)
        return cp

    def to_array(self):
        "Return the entries of the density matrix as a dense numpy ndarray."
        dense = np.triu(self.data.get())
        dense += np.conjugate(np.transpose(np.triu(dense, 1)))
        return dense

    def get_diag(self):
        diag = ga.empty((2**self.no_qubits), dtype=np.float64)
        block = (self._block_size, 1, 1)
        grid = (self._grid_size, 1, 1)

        _get_diag.prepared_call(grid, block,
                                self.data.gpudata, diag.gpudata, np.uint32(self.no_qubits))

        return diag.get()

    def cphase(self, bit0, bit1):
        assert bit0 < self.no_qubits
        assert bit1 < self.no_qubits

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _cphase.prepared_call(grid, block,
                              self.data.gpudata,
                              (1 << bit0) | (1 << bit1),
                              self.no_qubits)

    def hadamard(self, bit):
        assert bit < self.no_qubits

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _hadamard.prepared_call(grid, block,
                                self.data.gpudata,
                                1 << bit,
                                0.5,
                                self.no_qubits)

    def amp_ph_damping(self, bit, gamma, lamda):
        assert bit < self.no_qubits

        gamma = np.float64(gamma)
        lamda = np.float64(lamda)

        s1mgamma = np.float64(np.sqrt(1 - gamma))
        s1mlamda = np.float64(np.sqrt(1 - lamda))

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _amp_ph_damping.prepared_call(grid, block,
                                      self.data.gpudata,
                                      1 << bit,
                                      gamma, s1mgamma, s1mlamda,
                                      self.no_qubits)

    def rotate_y(self, bit, cosine, sine):
        assert bit < self.no_qubits

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _rotate_y.prepared_call(grid, block,
                                self.data.gpudata,
                                1 << bit,
                                cosine, sine,
                                self.no_qubits)

    def rotate_x(self, bit, cosine, sine):
        assert bit < self.no_qubits

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _rotate_x.prepared_call(grid, block,
                                self.data.gpudata,
                                1 << bit,
                                cosine, sine,
                                self.no_qubits)

    def rotate_z(self, bit, cosine2, sine2):
        assert bit < self.no_qubits

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _rotate_z.prepared_call(grid, block,
                                self.data.gpudata,
                                1 << bit,
                                cosine2, sine2,
                                self.no_qubits)

    def add_ancilla(self, bit, anc_st):
        assert bit <= self.no_qubits
        new_data = ga.zeros(
            (2**(self.no_qubits + 1), 2**(self.no_qubits + 1)),
            dtype=np.complex128)
        new_dm = Density(self.no_qubits + 1, new_data)

        dm9_0 = self.data
        dm9_1 = ga.zeros_like(self.data)
        if anc_st == 1:
            dm9_0, dm9_1 = dm9_1, dm9_0

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _dm_inflate.prepared_call(grid, block,
                                  new_dm.data.gpudata,
                                  bit,
                                  dm9_0.gpudata, dm9_1.gpudata,
                                  new_dm.no_qubits)

        return new_dm

    def measure_ancilla(self, bit):
        assert bit < self.no_qubits

        d0 = ga.zeros((2**(self.no_qubits - 1), 2**(self.no_qubits - 1)),
                      np.complex128)
        d1 = ga.zeros((2**(self.no_qubits - 1), 2**(self.no_qubits - 1)),
                      np.complex128)

        block = (self._block_size, self._block_size, 1)
        grid = (self._grid_size, self._grid_size, 1)

        _dm_reduce.prepared_call(grid, block,
                                 self.data.gpudata,
                                 bit,
                                 d0.gpudata, d1.gpudata,
                                 1, 1, self.no_qubits)

        dm0 = Density(self.no_qubits - 1, d0)
        dm1 = Density(self.no_qubits - 1, d1)

        p0 = dm0.trace()
        p1 = dm1.trace()
        return p0, dm0, p1, dm1
