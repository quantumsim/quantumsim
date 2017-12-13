# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as ga

from . import ptm
from . import dm_general_np

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
_multitake = mod.get_function("multitake")
_multitake.prepare("PPPPPPI")


class DensityGeneral:
    _gpuarray_cache = {}

    def __init__(self, bases, data=None):
        """
        Create a new density matrix for several qudits.

        bases: a list of ptm.PauliBasis describing the subsystems.

        data: a numpy.ndarray, gpuarray.array, or pycuda.driver.DeviceAllocation.
              must be of size (2**no_qubits, 2**no_qubits); is copied to GPU if not already there.
              Only upper triangle is relevant.
              If data is None, create a new density matrix with all qubits in ground state.
        """

        self.bases = bases

        shape = [pb.dim_pauli for pb in self.bases]

        if isinstance(data, ga.GPUArray):
            assert self.shape == data.shape
            self.data = data

        if data is None:
            self.data = np.zeros(shape, np.float64)
            ground_state_index = [pb.comp_basis_indices[0] for pb in self.bases]
            self.data[ground_state_index] = 1
            self.data = ga.to_gpu(self.data)
        else:
            raise ValueError("type of data not understood")


    def _cached_gpuarray(self, array):
        """
        Given a numpy array, 
        calculate the python hash of its bytes;
        
        If it is not found in the cache, upload to gpu
        and store in cache, otherwise return cached allocation.
        """
        key = hash(array.tobytes())
        try:
            array_gpu = self._gpuarray_cache[key]
        except KeyError:
            array_gpu = ga.to_gpu(array)
            self._gpuarray_cache[key] = array_gpu

        return array_gpu

    def trace(self):
        return np.sum(self.get_diag())

    def renormalize(self):
        """Renormalize to trace one."""
        tr = self.trace()
        self.data *= np.float(1 / tr)

    def copy(self):
        "Return a deep copy of this Density."
        data_cp = self.data.copy()
        cp = Density(self.bases, data=data_cp)
        return cp

    def to_array(self):
        "Return the entries of the density matrix as a dense numpy ndarray."
        host_dm = dm_general_np.DensityGeneralNP(self.bases, data=self.data.get())


    def get_diag(self, target_gpu_array=None, get_data=True):
        """
        Obtain the diagonal of the density matrix. 

        target_gpu_array: an already-allocated gpu array to which the data will be copied.
                          If None, make a new gpu array.

        get_data: boolean, whether the data should be copied from the gpu.
        """
        diag_size = pytools.product([len(pb.comp_basis_indices) for pb in self.bases])

        if target_gpu_array is None:
            target_gpu_array = ga.empty(diag_size, dtype=np.float64)
        else:
            assert target_gpu_array.size >= diag_size


        idx = [pb.comp_basis_indices for pb in self.pb]

        idx_j = np.array(list(pytools.flatten(idx))).astype(np.uint32)
        idx_i = np.cumsum([0]+[len(i) for i in idx][:-1]).astype(np.uint32)

        xshape = np.array(self.data.shape, np.uint32)
        yshape = np.array([pb.dim_hilbert for pb in self.bases], np.uint32)

        xshape_gpu = self._cached_gpuarray(xshape)
        yshape_gpu = self._cached_gpuarray(yshape)

        idx_i_gpu = self._cached_gpuarray(idx_i)
        idx_j_gpu = self._cached_gpuarray(idx_j)

        block = (2**8, 1, 1)
        grid = (2**max(0, self.no_qubits - 8), 1, 1)

        _multitake.prepared_call(
                grid,
                block,
                self.data.gpudata,
                self.target_gpu_array.gpudata,
                idx_i_gpu, idx_j_gpu,
                xshape_gpu, yshape_gpu,
                np.uint32(self.no_qubits)
                )

        if get_data:
            return target_gpu_array.get()
        else:
            return target_gpu_array

    def cphase(self, bit0, bit1):
        warnings.warn(
            "cphase deprecated, use two_ptm instead",
            DeprecationWarning)

        cphase_ptm = ptm.double_kraus_to_ptm(np.diag([1, 1, 1, -1])).real

        self.apply_two_ptm(bit0, bit1, cphase_ptm)

    def apply_two_ptm(self, bit0, bit1, ptm):
        assert bit0 < self.no_qubits
        assert bit1 < self.no_qubits

        ptm_gpu = self._cached_gpuarray(ptm)

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

        _single_qubit_ptm.prepared_call(grid,
                                        block,
                                        self.data.gpudata,
                                        ptm_gpu.gpudata,
                                        bit,
                                        self.no_qubits,
                                        shared_size=8 * (17 + self._blocksize))

    def add_ancilla(self, basis, state):
        subbasis = basis.get_subbasis([basis.comp_basis_indices[state]])
        self.bases.append(subbasis)

    def partial_trace(self, bit):
        """
        Return the diagonal of the reduced density matrix of a qubit.

        bit: integer index of the bit
        """

        assert bit < len(self.bases)

        # todo on graphics card, optimize for tracing out?
        diag = self.get_diag()

        diag.reshape([pb.dim_hilbert for pb in self.bases])

        in_indices = list(range(len(diag.shape)))
        out_indices = [bit]

        return np.einsum(diag, in_indices, bit)

    def project_measurement(self, bit, state, lazy_alloc=True):
        """
        Remove a qubit from the density matrix by projecting
        on a computational basis state.

        bit: which bit to project
        state: which state in the Hilbert space to project on
        lazy_alloc: bool. If True, do not allocate a smaller space
        for the new matrix, instead leave it at the same size as now,
        in anticipation of a future increase in size.
        """

        new_shape = self.data.shape
        del new_shape[bit]

        if self._work_data.size < new_size or not lazy_alloc:
            # force deallocation
            self._work_data.gpudata.free()
            # allocate new
            self._work_data = ga.empty(new_shape, np.float64)

        idx = []

        for i, pb in enumerate(self.bases):
            if i == bit:
                idx.append([pb.comp_basis_indices[state]])
            else:
                idx.append(list(range(pb.dim_pauli)))

        idx_j = np.array(list(pytools.flatten(idx))).astype(np.uint32)
        idx_i = np.cumsum([0]+[len(i) for i in idx][:-1]).astype(np.uint32)

        xshape = np.array(self.data.shape, np.uint32)
        yshape = np.array([pb.dim_hilbert for pb in self.bases], np.uint32)

        xshape_gpu = self._cached_gpuarray(xshape)
        yshape_gpu = self._cached_gpuarray(yshape)

        idx_i_gpu = self._cached_gpuarray(idx_i)
        idx_j_gpu = self._cached_gpuarray(idx_j)

        block = (2**8, 1, 1)
        grid = (2**max(0, self.no_qubits - 8), 1, 1)

        _multitake.prepared_call(
                grid,
                block,
                self.data.gpudata,
                self._work_data.gpudata,
                idx_i_gpu, idx_j_gpu,
                xshape_gpu, yshape_gpu,
                np.uint32(self.no_qubits)
                )

        self.data, self._work_data = self._work_data, self.data

        subbase_idx = [self.bases[bit].comp_basis_indices[state]]
        self.bases[bit] = self.bases[bit].get_subbasis(subbase_idx)

class DensityGeneralShim:
    """A subclass of Density that uses general_two_qubit_ptm as a backend,
    for testing purposes"""


    def apply_two_ptm(self, bit0, bit1, ptm):
        assert bit0 < self.no_qubits
        assert bit1 < self.no_qubits

        # bit0 must be the smaller one.
        if bit1 < bit0:
            bit1, bit0 = bit0, bit1
            ptm = np.einsum("abcd -> badc", ptm.reshape((4,4,4,4))).reshape((16,16))

        key = hash(ptm.tobytes())
        try:
            ptm_gpu = self._ptm_cache[key]
        except KeyError:
            assert ptm.shape == (16, 16)
            assert ptm.dtype == np.float64
            self._ptm_cache[key] = ga.to_gpu(ptm)
            ptm_gpu = self._ptm_cache[key]

        # dim_a_out, dim_b_out, d_internal (arbitrary)
        block = (4, 4, 16)
        blocksize = 4*4*16
        gridsize = max(1, (4**self.no_qubits)//blocksize)
        grid = (gridsize, 1, 1)


        _two_qubit_general_ptm.prepared_call(grid,
                                     block,
                                     self.data.gpudata,
                                     self.data.gpudata,
                                     ptm_gpu.gpudata,
                                     4, 4,
                                     4**bit0,
                                     4**(bit1-bit0-1),
                                     4**self.no_qubits,
                                     shared_size=8 * (256 + blocksize))
