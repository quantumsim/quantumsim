# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as ga

from . import ptm

import pycuda.autoinit

# load the kernels
from pycuda.compiler import SourceModule

import sys
import os

import warnings

package_path = os.path.dirname(os.path.realpath(__file__))

mod = None
for kernel_file in [
        sys.prefix +
        "/pycudakernels/primitives.cu",
        package_path +
        "/primitives.cu"]:
    try:
        with open(kernel_file, "r") as kernel_source_file:
            mod = SourceModule(
                kernel_source_file.read(), options=[
                    "--default-stream", "per-thread", "-lineinfo"])
            break
    except FileNotFoundError:
        pass

if mod is None:
    raise ImportError("could not find primitives.cu")

pycuda.autoinit.context.set_shared_config(
    drv.shared_config.EIGHT_BYTE_BANK_SIZE)


_cphase = mod.get_function("cphase")
_cphase.prepare("PIII")
_get_diag = mod.get_function("get_diag")
_get_diag.prepare("PPI")
_bit_to_pauli_basis = mod.get_function("bit_to_pauli_basis")
_bit_to_pauli_basis.prepare("PII")
_pauli_reshuffle = mod.get_function("pauli_reshuffle")
_pauli_reshuffle.prepare("PPII")
_single_qubit_ptm = mod.get_function("single_qubit_ptm")
_single_qubit_ptm.prepare("PPII")
_dm_reduce = mod.get_function("dm_reduce")
_dm_reduce.prepare("PIPII")
_trace = mod.get_function("trace")
_trace.prepare("Pi")

class Density:

    _ptm_cache = {}

    def __init__(self, no_qubits, data=None):
        """create a new density matrix for several qubits.
        no_qubits: number of qubits.
        data: a numpy.ndarray, gpuarray.array, or pycuda.driver.DeviceAllocation.
              must be of size (2**no_qubits, 2**no_qubits); is copied to GPU if not already there.
              Only upper triangle is relevant.
              If data is None, create a new density matrix with all qubits in ground state.
        """

        self.allocated_qubits = 0
        self._set_no_qubits(no_qubits)

        if no_qubits > 15:
            raise ValueError(
                "no_qubits=%d is way too many qubits, are you sure?" %
                no_qubits)

        if isinstance(data, np.ndarray):
            assert data.shape == (1 << no_qubits, 1 << no_qubits)
            data = data.astype(np.complex128)
            complex_dm = ga.to_gpu(data)
            block_size = 2**4
            grid_size = 2**max(no_qubits - 4, 0)
            grid = (grid_size, grid_size, 1)
            block = (block_size, block_size, 1)
            for i in range(self.no_qubits):
                _bit_to_pauli_basis.prepared_call(
                    grid, block, complex_dm.gpudata, 1 << i, self.no_qubits)

            self.data = ga.empty(self._size, np.float64)
            _pauli_reshuffle.prepared_call(
                grid, block, complex_dm.gpudata, self.data.gpudata, self.no_qubits, 0)
        elif isinstance(data, ga.GPUArray):
            assert data.size == self._size
            assert data.dtype == np.float64
            self.data = data
        elif data is None:
            d = np.zeros(self._size, np.float64)
            d[0] = 1
            self.data = ga.to_gpu(d)
        else:
            raise ValueError("type of data not understood")

    def _set_no_qubits(self, no_qubits):
        self.allocated_qubits = max(self.allocated_qubits, no_qubits)
        self.no_qubits = no_qubits

        self._size = 1 << (2 * no_qubits)
        self._blocksize = 2**8
        self._gridsize = 2**max(0, 2 * no_qubits - 8)

    def trace(self):

        if self.no_qubits > 10:
            raise NotImplementedError("Trace not implemented for more than 10 qubits yet")
        diag = ga.empty((2**self.no_qubits), dtype=np.float64)
        block = (2**self.no_qubits, 1, 1)
        grid = (1,1,1)

        _get_diag.prepared_call(
            grid,
            block,
            self.data.gpudata,
            diag.gpudata,
            np.uint32(self.no_qubits))

        _trace.prepared_call(grid, block,
                diag.gpudata, -1, shared_size=8*block[0])

        trace = diag[0].get()

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
        complex_dm = ga.zeros(
            (1 << self.no_qubits, 1 << self.no_qubits), np.complex128)
        block_size = 2**4
        grid_size = 2**max(self.no_qubits - 4, 0)
        grid = (grid_size, grid_size, 1)
        block = (block_size, block_size, 1)
        _pauli_reshuffle.prepared_call(
            grid, block, complex_dm.gpudata, self.data.gpudata, self.no_qubits, 1)
        for i in range(self.no_qubits):
            _bit_to_pauli_basis.prepared_call(
                grid, block, complex_dm.gpudata, 1 << i, self.no_qubits)

        return complex_dm.get()

    def get_diag(self):
        diag = ga.empty((1 << self.no_qubits), dtype=np.float64)
        block = (2**8, 1, 1)
        grid = (2**max(0, self.no_qubits - 8), 1, 1)

        _get_diag.prepared_call(
            grid,
            block,
            self.data.gpudata,
            diag.gpudata,
            np.uint32(self.no_qubits))

        return diag.get()

    def cphase(self, bit0, bit1):
        assert bit0 < self.no_qubits
        assert bit1 < self.no_qubits

        block = (self._blocksize, 1, 1)
        grid = (self._gridsize, 1, 1)

        _cphase.prepared_call(grid, block,
                              self.data.gpudata,
                              bit0, bit1,
                              self.no_qubits)

    def apply_ptm(self, bit, ptm):
        assert bit < self.no_qubits

        key = hash(ptm.tobytes())
        try:
            ptm_gpu = self._ptm_cache[key]
        except KeyError:
            assert ptm.shape == (4, 4)
            assert ptm.dtype == np.float64
            self._ptm_cache[key] = ga.to_gpu(ptm)
            ptm_gpu = self._ptm_cache[key]

        block = (self._blocksize, 1, 1)
        grid = (self._gridsize, 1, 1)

        _single_qubit_ptm.prepared_call(grid, block,
                                        self.data.gpudata, ptm_gpu.gpudata, bit, self.no_qubits,
                                        shared_size=8 * (17 + self._blocksize))

    def hadamard(self, bit):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.hadamard_ptm())

    def amp_ph_damping(self, bit, gamma, lamda):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.amp_ph_damping_ptm(gamma, lamda))

    def rotate_y(self, bit, angle):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.rotate_y_ptm(angle))

    def rotate_x(self, bit, angle):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.rotate_x_ptm(angle))

    def rotate_z(self, bit, angle):
        warnings.warn("use apply_ptm")
        self.apply_ptm(bit, ptm.rotate_z_ptm(angle))

    def add_ancilla(self, anc_st):
        """Add an ancilla in the ground or excited state as the highest new bit.
        """

        new_dm = ga.zeros(self._size * 4, np.float64)

        offset = anc_st * (0x3 << (2 * self.no_qubits)) * 8

        drv.memcpy_dtod(int(new_dm.gpudata) + offset,
                        self.data.gpudata, self.data.nbytes)

        return Density(self.no_qubits + 1, new_dm)

    def partial_trace(self, bit):
        assert bit < self.no_qubits
        if self.no_qubits > 10:
            raise NotImplementedError("Trace not implemented for more than 10 qubits yet")
        diag = ga.empty((2**self.no_qubits), dtype=np.float64)
        block = (2**self.no_qubits, 1, 1)
        grid = (1,1,1)

        _get_diag.prepared_call(
            grid,
            block,
            self.data.gpudata,
            diag.gpudata,
            np.uint32(self.no_qubits))


        _trace.prepared_call(grid, block,
                diag.gpudata, bit, shared_size=8*block[0])

        tr1 = diag[0].get()
        tr0 = diag[1].get()

        return tr0, tr1

    def project_measurement(self, bit, state):
        assert bit < self.no_qubits

        d_new = ga.empty(self._size >> 2, np.float64)
        block = (self._blocksize, 1, 1)
        grid = (self._gridsize, 1, 1)

        _dm_reduce.prepared_call(grid, block,
                                 self.data.gpudata,
                                 bit,
                                 d_new.gpudata, state, self.no_qubits)

        return Density(self.no_qubits-1, d_new)

