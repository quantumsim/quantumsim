# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as ga

import pycuda.autoinit

# load the kernels
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS

import sys
import os

from pytools import product

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
                kernel_source_file.read(), options=DEFAULT_NVCC_FLAG+[
                    "--default-stream", "per-thread", "-lineinfo"])
            break
    except FileNotFoundError:
        pass

if mod is None:
    raise ImportError("could not find primitives.cu")

pycuda.autoinit.context.set_shared_config(
    drv.shared_config.EIGHT_BYTE_BANK_SIZE)

_two_qubit_ptm = mod.get_function("two_qubit_ptm")
_two_qubit_ptm.prepare("PPIII")
# _cphase = mod.get_function("cphase")
# _cphase.prepare("PIII")
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
_swap = mod.get_function("swap")
_swap.prepare("PIII")
_two_qubit_general_ptm = mod.get_function("two_qubit_general_ptm")
_two_qubit_general_ptm.prepare("PPPIIIII")


class Density:

    _ptm_cache = {}

    def __init__(self, dimensions, data=None):
        """A density matrix for several subsystems, where the dimension of each subsystem can be chosen.

        dimensions = list of ints, dimension. len(dimensions) is the number of qubits.

        data: a numpy.ndarray, gpuarray.array, or pycuda.driver.DeviceAllocation.
              must be of size (2**no_qubits, 2**no_qubits); is copied to GPU if not already there.
              Only upper triangle is relevant.
              If data is None, create a new density matrix with all qubits in ground state.
        """

        # two gpuarrays to allocate regions on the GPU
        # data holds the current data, _next_data is used as
        # working buffer and is not guaranteed to be anything

        self.data = None
        self._next_data = None

        # current dimensions (of data)
        self.dimensions = dimensions

        # for now, we choose a tight layout
        self._size = product(self.dimensions)

        if len(self.dimensions) > 15:
            raise ValueError(
                "no_qubits=%d is way too many qubits, are you sure?" %
                self.no_qubits)

        if data is None:
            # think about whether this is the gs in general...
            d = np.zeros(self._size, np.float64).reshape(self.dimensions)
            d[0] = 1
            self.data = ga.to_gpu(d)
        else:
            raise ValueError("type of data not understood")

    def trace(self):
        # wow I need to think
        raise NotImplementedError()

    def renormalize(self):
        """Renormalize to trace one."""
        tr = self.trace()
        self.data *= np.float(1 / tr)

    def copy(self):
        "Return a deep copy of this Density."
        data_cp = self.data.copy()
        cp = Density(self.dimensions, data=data_cp)
        return cp

    def to_array(self):
        "Return the entries of the density matrix as a dense numpy ndarray."

        return self.data.get()

    def get_diag(self):
        if self.allocated_diag < self.no_qubits:
            self.diag_work = ga.empty((1 << self.no_qubits), dtype=np.float64)
            self.allocated_qubits = self.no_qubits

        block = (2**8, 1, 1)
        grid = (2**max(0, self.no_qubits - 8), 1, 1)

        _get_diag.prepared_call(
            grid,
            block,
            self.data.gpudata,
            self.diag_work.gpudata,
            np.uint32(self.no_qubits))

        return self.diag_work.get()


    def multi_basis_project(self, selector):
        """
        Perform a projection to a cartesian subbasis on many
        axes at the same time. (Essentially slicing).

        Generalization of the get_diag method.
        """




    def apply_two_ptm(self, bit0, bit1, ptm):
        assert bit0 < self.no_qubits
        assert bit1 < self.no_qubits

        key = hash(ptm.tobytes())
        try:
            ptm_gpu = self._ptm_cache[key]
        except KeyError:
            assert ptm.shape == (16, 16)
            assert ptm.dtype == np.float64
            self._ptm_cache[key] = ga.to_gpu(ptm)
            ptm_gpu = self._ptm_cache[key]

        block = (self._blocksize, 1, 1)
        grid = (self._gridsize, 1, 1)

        _two_qubit_ptm.prepared_call(grid,
                                     block,
                                     self.data.gpudata,
                                     ptm_gpu.gpudata,
                                     bit0,
                                     bit1,
                                     self.no_qubits,
                                     shared_size=8 * (256 + self._blocksize))

    def add_ancilla(self, anc_st, anc_dim):
        """Add an ancilla in the ground or excited state as the highest new bit.
        """

        byte_size_of_smaller_dm = 2**(2 * self.no_qubits) * 8

        if self.allocated_qubits == self.no_qubits:
            # allocate larger memory
            new_dm = ga.zeros(self._size * 4, np.float64)
            offset = anc_st * 3 * byte_size_of_smaller_dm
            drv.memcpy_dtod(int(new_dm.gpudata) + offset,
                            self.data.gpudata, byte_size_of_smaller_dm)

            self.data = new_dm
        else:
            # reuse previously allocated memory
            if anc_st == 0:
                drv.memset_d8(int(self.data.gpudata) + byte_size_of_smaller_dm,
                              0, 3 * byte_size_of_smaller_dm)
            if anc_st == 1:
                drv.memcpy_dtod(int(self.data.gpudata) + 3 * byte_size_of_smaller_dm,
                                self.data.gpudata, byte_size_of_smaller_dm)
                drv.memset_d8(self.data.gpudata, 0, 3 *
                              byte_size_of_smaller_dm)

        self._set_no_qubits(self.no_qubits + 1)

    def partial_trace(self, bit):
        assert bit < self.no_qubits
        if self.no_qubits > 10:
            raise NotImplementedError(
                "Trace not implemented for more than 10 qubits yet")
        if self.allocated_diag < self.no_qubits:
            self.diag_work = ga.empty((1 << self.no_qubits), dtype=np.float64)
            self.allocated_qubits = self.no_qubits
        block = (2**self.no_qubits, 1, 1)
        grid = (1, 1, 1)

        _get_diag.prepared_call(
            grid,
            block,
            self.data.gpudata,
            self.diag_work.gpudata,
            np.uint32(self.no_qubits))

        _trace.prepared_call(
            grid,
            block,
            self.diag_work.gpudata,
            bit,
            shared_size=8 *
            block[0])

        tr1, tr0 = self.diag_work[:2].get()
        return tr0, tr1

    def project_measurement(self, bit, state):
        assert bit < self.no_qubits

        block = (self._blocksize, 1, 1)
        grid = (self._gridsize, 1, 1)

        if bit != self.no_qubits - 1:
            _swap.prepared_call(grid, block,
                                self.data.gpudata,
                                bit,
                                self.no_qubits - 1,
                                self.no_qubits)

        if state == 1:
            byte_size_of_smaller_dm = 2**(2 * self.no_qubits - 2) * 8
            drv.memcpy_dtod(self.data.gpudata,
                            int(self.data.gpudata) + 3 *
                            byte_size_of_smaller_dm,
                            byte_size_of_smaller_dm)

        self._set_no_qubits(self.no_qubits - 1)

        return self
