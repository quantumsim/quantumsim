# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as ga

import pytools

from . import ptm
from . import dm_general_np

import pycuda.autoinit
import pycuda.tools

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

        shape = tuple([pb.dim_pauli for pb in self.bases])

        if isinstance(data, ga.GPUArray):
            assert shape == data.shape
            self.data = data
        elif data is None:
            self.data = np.zeros(shape, np.float64)
            ground_state_index = [pb.comp_basis_indices[0] for pb in self.bases]
            self.data[tuple(ground_state_index)] = 1
            self.data = ga.to_gpu(self.data)
        else:
            raise ValueError("type of data not understood")

        self.data.gpudata.size = self.data.nbytes
        self._work_data = ga.empty_like(self.data)
        self._work_data.gpudata.size = self._work_data.nbytes

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
        # todo there is a smarter way of doing this with pauli-dirac basis
        return np.sum(self.get_diag())

    def renormalize(self):
        """Renormalize to trace one."""
        tr = self.trace()
        self.data *= np.float(1 / tr)

    def copy(self):
        "Return a deep copy of this Density."
        data_cp = self.data.copy()
        cp = DensityGeneral(self.bases, data=data_cp)
        return cp

    def to_array(self):
        "Return the entries of the density matrix as a dense numpy ndarray."
        dimensions = [2]*self.no_qubits

        host_dm = dm_general_np.DensityGeneralNP(dimensions, 
                data=self.data.get()).to_array()

        return host_dm

    def get_diag(self, target_gpu_array=None, get_data=True):
        """
        Obtain the diagonal of the density matrix. 

        target_gpu_array: an already-allocated gpu array to which the data will be copied.
                          If None, make a new gpu array.

        get_data: boolean, whether the data should be copied from the gpu.
        """

        diag_bases = [pb.get_classical_subbasis() for pb in self.bases]
        diag_shape = [db.dim_pauli for db in diag_bases]
        diag_size = pytools.product(diag_shape)

        if target_gpu_array is None:
            if self._work_data.gpudata.size < diag_size*8:
                self._work_data.gpudata.free()
                self._work_data = ga.empty(diag_shape, np.float64)
                self._work_data.gpudata.size = self._work_data.nbytes
            target_gpu_array = self._work_data
        else:
            assert target_gpu_array.size >= diag_size

        idx = [[pb.comp_basis_indices[i] 
                for i in range(pb.dim_hilbert)
                if pb.comp_basis_indices[i] is not None]
                for pb in self.bases
                ]

        idx_j = np.array(list(pytools.flatten(idx))).astype(np.uint32)
        idx_i = np.cumsum([0]+[len(i) for i in idx][:-1]).astype(np.uint32)

        xshape = np.array(self.data.shape, np.uint32)
        yshape = np.array(diag_shape, np.uint32)

        xshape_gpu = self._cached_gpuarray(xshape)
        yshape_gpu = self._cached_gpuarray(yshape)

        idx_i_gpu = self._cached_gpuarray(idx_i)
        idx_j_gpu = self._cached_gpuarray(idx_j)

        block = (2**8, 1, 1)
        grid = (max(1, (diag_size-1)//2**8 + 1), 1, 1)

        if len(yshape) == 0:
            # brain-dead case, but should be handled according to exp.
            target_gpu_array.set(self.data.get())
        else:
            _multitake.prepared_call(
                    grid,
                    block,
                    self.data.gpudata,
                    target_gpu_array.gpudata,
                    idx_i_gpu.gpudata, idx_j_gpu.gpudata,
                    xshape_gpu.gpudata, yshape_gpu.gpudata,
                    np.uint32(len(yshape))
                    )

        if get_data:
            return target_gpu_array.get().ravel()[:diag_size]
        else:
            return target_gpu_array

    def cphase(self, bit0, bit1):
        warnings.warn(
            "cphase deprecated, use two_ptm instead",
            DeprecationWarning)

        cphase_ptm = ptm.double_kraus_to_ptm(np.diag([1, 1, 1, -1])).real

        self.apply_two_ptm(bit0, bit1, cphase_ptm)

    def apply_two_ptm(self, bit0, bit1, ptm):
        """
        Apply a two-qubit Pauli transfer matrix to qubit bit0 and bit1.

        ptm: np.array, a two-qubit ptm in the basis of bit0 and bit1
        bit0, bit1: integer indices

        So far only works for square ptms, and thus is done in-place
        """
        assert 0 <= bit0 < len(self.bases)
        assert 0 <= bit1 < len(self.bases)

        # bit0 must be the smaller one.
        if bit1 < bit0:
            bit1, bit0 = bit0, bit1
            ptm = np.einsum("abcd -> badc", ptm.reshape((4,4,4,4))).reshape((16,16))

        dim0 = self.bases[bit0].dim_pauli
        dim1 = self.bases[bit1].dim_pauli

        ptm_gpu = self._cached_gpuarray(ptm)

        # dim_a_out, dim_b_out, d_internal (arbitrary)
        block = (dim0, dim1, 16)
        blocksize = dim0*dim1*16
        gridsize = max(1, (self.data.size-1)//blocksize+1)
        grid = (gridsize, 1, 1)

        dim_z = pytools.product(self.data.shape[0:bit0])
        dim_y = pytools.product(self.data.shape[bit0+1:bit1])
        dim_rho = self.data.size

        _two_qubit_general_ptm.prepared_call(grid,
                                     block,
                                     self.data.gpudata,
                                     self.data.gpudata,
                                     ptm_gpu.gpudata,
                                     dim0, dim1,
                                     dim_z,
                                     dim_y,
                                     dim_rho,
                                     shared_size=8 * (ptm.size + blocksize)
                                     )



    def apply_ptm(self, bit, ptm):
        """
        Apply a one-qubit Pauli transfer matrix to qubit bit.

        ptm: np.array, a ptm in the basis of bit.
        bit: integer qubit index

        So far only works for square ptms, and thus is done in-place
        """

        assert 0 <= bit < len(self.bases)

        dim_bit = self.bases[bit].dim_pauli

        ptm_gpu = self._cached_gpuarray(ptm)

        dint = min(64, self.data.size//dim_bit)
        block = (1, dim_bit, dint)
        blocksize = dim_bit*dint
        gridsize = max(1, (self.data.size-1)//blocksize+1)
        grid = (gridsize, 1, 1)

        dim_z = pytools.product(self.data.shape[bit+1:])
        dim_y = pytools.product(self.data.shape[:bit])
        dim_rho = self.data.size

        _two_qubit_general_ptm.prepared_call(grid,
                                     block,
                                     self.data.gpudata,
                                     self.data.gpudata,
                                     ptm_gpu.gpudata,
                                     1, dim_bit,
                                     dim_z,
                                     dim_y,
                                     dim_rho,
                                     shared_size=8 * (ptm.size + blocksize)
                                     )

    def add_ancilla(self, basis, state):
        """
        add an ancilla with `basis` and with state state.
        """

        # figure out the projection matrix
        ptm = np.zeros(basis.dim_pauli)
        ptm[basis.comp_basis_indices[state]] = 1

        # make sure work_data is large enough, reshape it

        # TODO hacky as fuck: we put the allocated size 
        # into the allocation object by hand

        new_shape = tuple([basis.dim_pauli] + list(self.data.shape))
        new_size_bytes = pytools.product(new_shape)*8

        if self._work_data.gpudata.size < new_size_bytes:
            # reallocate
            self._work_data.gpudata.free()
            self._work_data = ga.empty(new_shape, np.float64)
            self._work_data.gpudata.size = self._work_data.nbytes
        else:
            # reallocation not required, 
            # reshape but reeuse allocation
            self._work_data = ga.GPUArray(
                    shape=new_shape,
                    dtype=np.float64,
                    gpudata=self._work_data.gpudata,
                    )

        # perform projection
        ptm_gpu = self._cached_gpuarray(ptm)

        dim_bit = basis.dim_pauli
        dint = min(64, new_size_bytes//8//dim_bit)
        block = (1, dim_bit, dint)
        blocksize = dim_bit*dint
        gridsize = max(1, (new_size_bytes//8-1)//blocksize+1)
        grid = (gridsize, 1, 1)

        dim_z = self.data.size #1
        dim_y = 1
        dim_rho = self.data.size


        _two_qubit_general_ptm.prepared_call(grid,
                                     block,
                                     self.data.gpudata,
                                     self._work_data.gpudata,
                                     ptm_gpu.gpudata,
                                     1, 1,
                                     dim_z,
                                     dim_y,
                                     dim_rho,
                                     shared_size=8 * (ptm.size + blocksize)
                                     )

        self.data, self._work_data = self._work_data, self.data
        self.bases = [basis] + self.bases



    def partial_trace(self, bit):
        """
        Return the diagonal of the reduced density matrix of a qubit.

        bit: integer index of the bit
        """

        assert 0 <= bit < len(self.bases)

        # todo on graphics card, optimize for tracing out?
        diag = self.get_diag()

        diag = diag.reshape([pb.dim_hilbert for pb in self.bases])

        in_indices = list(range(len(diag.shape)))
        out_indices = [bit]

        return np.einsum(diag, in_indices, [bit])

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

        assert 0 <= bit < len(self.bases)

        new_shape = list(self.data.shape)
        new_shape[bit] = 1

        new_size_bytes = self.data.nbytes//self.bases[bit].dim_pauli

        # TODO hacky as fuck: put the allocated size into the allocation by hand
        if self._work_data.gpudata.size < new_size_bytes or not lazy_alloc:
            # reallocate
            self._work_data.gpudata.free()
            self._work_data = ga.empty(new_shape, np.float64)
            self._work_data.gpudata.size = self._work_data.nbytes
        else:
            # reallocation not required, 
            # reshape but reeuse allocation
            self._work_data = ga.GPUArray(
                    shape=new_shape,
                    dtype=np.float64,
                    gpudata=self._work_data.gpudata,
                    )

        idx = []
        # todo: can be built more efficiently

        for i, pb in enumerate(self.bases):
            if i == bit:
                idx.append([pb.comp_basis_indices[state]])
            else:
                idx.append(list(range(pb.dim_pauli)))


        idx_j = np.array(list(pytools.flatten(idx))).astype(np.uint32)
        idx_i = np.cumsum([0]+[len(i) for i in idx][:-1]).astype(np.uint32)

        xshape = np.array(self.data.shape, np.uint32)
        yshape = np.array(new_shape, np.uint32)

        xshape_gpu = self._cached_gpuarray(xshape)
        yshape_gpu = self._cached_gpuarray(yshape)

        idx_i_gpu = self._cached_gpuarray(idx_i)
        idx_j_gpu = self._cached_gpuarray(idx_j)

        block = (2**8, 1, 1)
        grid = (max(1, (self._work_data.size-1)//2**8 + 1), 1, 1)

        _multitake.prepared_call(
                grid,
                block,
                self.data.gpudata,
                self._work_data.gpudata,
                idx_i_gpu.gpudata, idx_j_gpu.gpudata,
                xshape_gpu.gpudata, yshape_gpu.gpudata,
                np.uint32(len(xshape))
                )

        self.data, self._work_data = self._work_data, self.data

        subbase_idx = [self.bases[bit].comp_basis_indices[state]]
        self.bases[bit] = self.bases[bit].get_subbasis(subbase_idx)

class DensityGeneralShim(DensityGeneral):
    """A subclass of Density that uses general_two_qubit_ptm as a backend,
    for testing purposes"""
    pb = ptm.PauliBasis_0xy1()

    def __init__(self, no_qubits, data=None):

        if isinstance(data, np.ndarray):
            from . import dm_np
            dm_cpu = dm_np.DensityNP(no_qubits, data)
            data = dm_cpu.dm
            data = ga.to_gpu(data)

        super().__init__([self.pb]*no_qubits, data)

    def to_array(self):
        "Return the entries of the density matrix as a dense numpy ndarray."
        from . import dm_np

        dm = dm_np.DensityNP(self.no_qubits) 
        # transpose switches order: in new dm, qubit 0 is msb, in old it is lsb
        dm.dm = self.data.get().squeeze()
        
        return dm.to_array()

    def copy(self):
        "Return a deep copy of this Density."
        data_cp = self.data.copy()
        cp = self.__class__(self.no_qubits, data=data_cp)
        return cp

    def add_ancilla(self, st):
        super().add_ancilla(self.pb, st)

    @property
    def no_qubits(self):
        # ignore bases of length 1 (new feature)
        b = [b for b in self.bases if b.dim_pauli == 4]
        return len(b)

    def translate_bit(self, bit):
        # old style bits are labelled 0 -> lsb,
        # new style bits are labelled 0 -> msb, 
        # and we ignore bases of length 1

        assert 0 <= bit < self.no_qubits

        alive_idx = [i for i, pb in enumerate(self.bases)
                if pb.dim_pauli == 4]

        return self.no_qubits - alive_idx[bit] - 1
        
    def cphase(self, bit0, bit1):
        bit0 = self.translate_bit(bit0)
        bit1 = self.translate_bit(bit1)
        cphase_ptm = ptm.double_kraus_to_ptm(np.diag([1, 1, 1, -1])).real
        self.apply_two_ptm(bit0, bit1, cphase_ptm)

    def hadamard(self, bit):
        bit = self.translate_bit(bit)
        self.apply_ptm(bit, ptm.hadamard_ptm())

    def amp_ph_damping(self, bit, gamma, lamda):
        bit = self.translate_bit(bit)
        self.apply_ptm(bit, ptm.amp_ph_damping_ptm(gamma, lamda))

    def rotate_y(self, bit, angle):
        bit = self.translate_bit(bit)
        self.apply_ptm(bit, ptm.rotate_y_ptm(angle))

    def rotate_x(self, bit, angle):
        bit = self.translate_bit(bit)
        self.apply_ptm(bit, ptm.rotate_x_ptm(angle))

    def rotate_z(self, bit, angle): 
        bit = self.translate_bit(bit)
        self.apply_ptm(bit, ptm.rotate_z_ptm(angle))

    def project_measurement(self, bit, state):
        bit = self.translate_bit(bit)
        super().project_measurement(bit, state)

    def partial_trace(self, bit):
        bit = self.translate_bit(bit)
        return super().partial_trace(bit)
