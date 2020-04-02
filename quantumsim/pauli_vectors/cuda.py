# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt
import sys

import numpy as np
import os
import pytools
import warnings

# noinspection PyUnresolvedReferences
import pycuda.autoinit
# noinspection PyUnresolvedReferences
import pycuda.driver as drv
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as ga
# noinspection PyUnresolvedReferences
import pycuda.reduction
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS

from .pauli_vector import PauliVectorBase

package_path = os.path.dirname(os.path.realpath(__file__))

mod = None

for kernel_file in [sys.prefix + "/pycudakernels/primitives.cu",
                    package_path + "/primitives.cu"]:
    try:
        with open(kernel_file, "r") as kernel_source_file:
            mod = SourceModule(
                kernel_source_file.read(), options=DEFAULT_NVCC_FLAGS + [
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

sum_along_axis = pycuda.reduction.ReductionKernel(
    dtype_out=np.float64,
    neutral="0", reduce_expr="a+b",
    map_expr="(i/stride) % dim == offset ? in[i] : 0",
    arguments="const double *in, unsigned int stride, unsigned int dim, "
              "unsigned int offset"
)


class PauliVectorCuda(PauliVectorBase):
    _gpuarray_cache = {}

    def __init__(self, bases, pv=None, *, force=False):
        """Create a new density matrix for several qudits.

        Parameters
        ----------
        bases : list of quantumsim.bases.PauliBasis
            Dimensions of qubits in the system.

        pv : array or None.
            Must be of size (2**no_qubits, 2**no_qubits). Only upper triangle
            is relevant.  If data is `None`, create a new density matrix with
            all qubits in ground state.
        """
        super().__init__(bases, pv, force=force)
        if pv is not None:
            if self.dim_pauli != pv.shape:
                raise ValueError(
                    '`bases` Pauli dimensionality should be the same as the '
                    'shape of `data` array.\n'
                    ' - bases shapes: {}\n - data shape: {}'
                    .format(self.dim_pauli, pv.shape))
        else:
            pv = np.zeros(self.dim_pauli, np.float64)
            ground_state_index = [pb.computational_basis_indices[0]
                                  for pb in self.bases]
            pv[tuple(ground_state_index)] = 1

        if isinstance(pv, np.ndarray):
            if pv.dtype not in (np.float16, np.float32, np.float64):
                raise ValueError(
                    '`pv` must have float64 data type, got {}'
                    .format(pv.dtype)
                )

            # Looks like there are some issues with ordering, so the line
            # below per se does not work.
            # self._data = ga.to_gpu(pv.astype(np.float64))

            self._work_data = ga.to_gpu(
                pv.reshape(pv.size, order='C').astype(np.float64))
            self._data = ga.empty(pv.shape, dtype=np.float64, order='C')
            self._data.set(self._work_data.reshape(pv.shape))
            self._work_data.gpudata.free()
        elif isinstance(pv, ga.GPUArray):
            if pv.dtype != np.float64:
                raise ValueError(
                    '`pv` must have float64 data type, got {}'
                    .format(pv.dtype)
                )
            self._data = pv
        else:
            raise ValueError(
                "`pv` must be Numpy array, PyCUDA GPU array or "
                "None, got type `{}`".format(type(pv)))

        self._data.gpudata.size = self._data.nbytes
        self._work_data = ga.empty_like(self._data)
        self._work_data.gpudata.size = self._work_data.nbytes

    def to_pv(self):
        return self._data.get()

    def apply_ptm(self, ptm, *qubits):
        if len(qubits) == 1:
            self._apply_single_qubit_ptm(qubits[0], ptm)
        elif len(qubits) == 2:
            self._apply_two_qubit_ptm(qubits[0], qubits[1], ptm)
        else:
            raise NotImplementedError('Applying {}-qubit PTM is not '
                                      'implemented in the active backend.')

    def _apply_two_qubit_ptm(self, qubit0, qubit1, ptm):
        """Apply a two-qubit Pauli transfer matrix to qubit `bit0` and `bit1`.

        Parameters
        ----------
        ptm: array-like
            A two-qubit ptm in the basis of `bit0` and `bit1`. Must be a 4D
            matrix with dimensions, that correspond to the qubits.
        qubit1 : int
            Index of first qubit
        qubit0: int
            Index of second qubit
        """
        self._validate_qubit(qubit1, 'qubit0')
        self._validate_qubit(qubit0, 'qubit1')
        if len(ptm.shape) != 4:
            raise ValueError(
                "`ptm` must be a 4D array, got {}D".format(len(ptm.shape)))

        # bit0 must be the more significant bit (bit 0 is msb)
        if qubit0 > qubit1:
            qubit0, qubit1 = qubit1, qubit0
            ptm = np.einsum("abcd -> badc", ptm)

        new_shape = list(self._data.shape)
        dim0_out, dim1_out, dim0_in, dim1_in = ptm.shape
        assert new_shape[qubit1] == dim1_in
        assert new_shape[qubit0] == dim0_in
        new_shape[qubit1] = dim1_out
        new_shape[qubit0] = dim0_out
        new_size = pytools.product(new_shape)
        new_size_bytes = new_size * 8

        if self._work_data.gpudata.size < new_size_bytes:
            # reallocate
            self._work_data.gpudata.free()
            self._work_data = ga.empty(new_shape, np.float64)
            self._work_data.gpudata.size = self._work_data.nbytes
        else:
            # reallocation not required,
            # reshape but reuse allocation
            self._work_data = ga.GPUArray(
                shape=new_shape,
                dtype=np.float64,
                gpudata=self._work_data.gpudata,
            )

        ptm_gpu = self._cached_gpuarray(ptm)

        rest_shape = new_shape.copy()
        rest_shape[qubit1] = 1
        rest_shape[qubit0] = 1

        dint = 1
        for i in sorted(rest_shape):
            if i * dint > 256 // (dim0_out * dim1_out):
                break
            else:
                dint *= i

        # dim_a_out, dim_b_out, d_internal (arbitrary)
        block = (dim0_out, dim1_out, dint)
        blocksize = dim1_out * dim0_out * dint
        sh_mem_size = dint * dim1_in * dim0_in  # + ptm.size
        grid_size = max(1, (new_size - 1) // blocksize + 1)
        grid = (grid_size, 1, 1)

        dim_z = pytools.product(self._data.shape[qubit1 + 1:])
        dim_y = pytools.product(self._data.shape[qubit0 + 1:qubit1])
        dim_rho = new_size  # self.data.size

        _two_qubit_general_ptm.prepared_call(
            grid,
            block,
            self._data.gpudata,
            self._work_data.gpudata,
            ptm_gpu.gpudata,
            dim0_in, dim1_in,
            dim_z,
            dim_y,
            dim_rho,
            shared_size=8 * sh_mem_size)

        self._data, self._work_data = self._work_data, self._data

    def _apply_single_qubit_ptm(self, qubit, ptm):
        # noinspection PyUnresolvedReferences
        """Apply a one-qubit Pauli transfer matrix to qubit bit.

        Parameters
        ----------
        qubit: int
            Qubit index
        ptm: array-like
            A PTM in the basis of a qubit.
        basis_out: quantumsim.bases.PauliBasis or None
            If provided, will convert qubit basis to specified
            after the PTM application.
        """
        new_shape = list(self._data.shape)
        self._validate_qubit(qubit, 'bit')

        # TODO Refactor to use self._validate_ptm
        if len(ptm.shape) != 2:
            raise ValueError(
                "`ptm` must be a 2D array, got {}D".format(len(ptm.shape)))

        dim_bit_out, dim_bit_in = ptm.shape
        new_shape[qubit] = dim_bit_out
        assert new_shape[qubit] == dim_bit_out
        new_size = pytools.product(new_shape)
        new_size_bytes = new_size * 8

        if self._work_data.gpudata.size < new_size_bytes:
            # reallocate
            self._work_data.gpudata.free()
            self._work_data = ga.empty(new_shape, np.float64)
            self._work_data.gpudata.size = self._work_data.nbytes
        else:
            # reallocation not required,
            # reshape but reuse allocation
            self._work_data = ga.GPUArray(
                shape=new_shape,
                dtype=np.float64,
                gpudata=self._work_data.gpudata,
            )

        ptm_gpu = self._cached_gpuarray(ptm)

        dint = min(64, self._data.size // dim_bit_in)
        block = (1, dim_bit_out, dint)
        blocksize = dim_bit_out * dint
        grid_size = max(1, (new_size - 1) // blocksize + 1)
        grid = (grid_size, 1, 1)

        dim_z = pytools.product(self._data.shape[qubit + 1:])
        dim_y = pytools.product(self._data.shape[:qubit])
        dim_rho = new_size  # self.data.size

        _two_qubit_general_ptm.prepared_call(
            grid,
            block,
            self._data.gpudata,
            self._work_data.gpudata,
            ptm_gpu.gpudata,
            1, dim_bit_in,
            dim_z,
            dim_y,
            dim_rho,
            shared_size=8 * (ptm.size + blocksize))

        self._data, self._work_data = self._work_data, self._data

    def diagonal(self, *, get_data=True, target_array=None, flatten=True):
        """Obtain the diagonal of the density matrix.

        Parameters
        ----------
        target_array : None or pycuda.gpuarray.array
            An already-allocated GPU array to which the data will be copied.
            If `None`, make a new GPU array.
        get_data : boolean
            Whether the data should be copied from the GPU.
        flatten : boolean
            TODO docstring
        """
        diag_bases = [pb.computational_subbasis() for pb in self.bases]
        diag_shape = [db.dim_pauli for db in diag_bases]
        diag_size = pytools.product(diag_shape)

        if target_array is None:
            if self._work_data.gpudata.size < diag_size * 8:
                self._work_data.gpudata.free()
                self._work_data = ga.empty(diag_shape, np.float64)
                self._work_data.gpudata.size = self._work_data.nbytes
            target_array = self._work_data
        else:
            if target_array.size < diag_size:
                raise ValueError(
                    "Size of `target_gpu_array` is too small ({}).\n"
                    "Should be at least {}."
                    .format(target_array.size, diag_size))

        idx = [[pb.computational_basis_indices[i]
                for i in range(pb.dim_hilbert)
                if pb.computational_basis_indices[i] is not None]
               for pb in self.bases]

        idx_j = np.array(list(pytools.flatten(idx))).astype(np.uint32)
        idx_i = np.cumsum([0] + [len(i) for i in idx][:-1]).astype(np.uint32)

        xshape = np.array(self._data.shape, np.uint32)
        yshape = np.array(diag_shape, np.uint32)

        xshape_gpu = self._cached_gpuarray(xshape)
        yshape_gpu = self._cached_gpuarray(yshape)

        idx_i_gpu = self._cached_gpuarray(idx_i)
        idx_j_gpu = self._cached_gpuarray(idx_j)

        block = (2 ** 8, 1, 1)
        grid = (max(1, (diag_size - 1) // 2 ** 8 + 1), 1, 1)

        if len(yshape) == 0:
            # brain-dead case, but should be handled according to exp.
            target_array.set(self._data.get())
        else:
            _multitake.prepared_call(
                grid, block, self._data.gpudata, target_array.gpudata,
                idx_i_gpu.gpudata, idx_j_gpu.gpudata,
                xshape_gpu.gpudata, yshape_gpu.gpudata,
                np.uint32(len(yshape))
            )

        if get_data:
            if flatten:
                return target_array.get().ravel()[:diag_size]
            else:
                return (target_array.get().ravel()[:diag_size]
                        .reshape(diag_shape))
        else:
            return ga.GPUArray(shape=diag_shape,
                               gpudata=target_array.gpudata,
                               dtype=np.float64)

    def trace(self):
        # TODO: there is a smarter way of doing this with pauli-dirac basis
        return np.sum(self.diagonal())

    def partial_trace(self, *qubits):
        raise NotImplementedError("Currently this method is implemented only "
                                  "in Numpy backend.")

    def meas_prob(self, qubit):
        """ Return the diagonal of the reduced density matrix of a qubit.

        Parameters
        ----------
        qubit: int
            Index of the qubit.
        """
        self._validate_qubit(qubit, 'qubit')

        # TODO on graphics card, optimize for tracing out?
        diag = self.diagonal(get_data=False)

        res = []
        stride = diag.strides[qubit] // 8
        dim = diag.shape[qubit]
        for offset in range(dim):
            pt = sum_along_axis(diag, stride, dim, offset)
            res.append(pt)

        out = [p.get().item() for p in res]
        if len(out) == self.dim_hilbert:
            return out
        else:
            # We need to insert zeros at the basis elements, that are missing
            # from the basis
            it = iter(out)
            return [next(it) if qbi is not None else 0.
                    for qbi in self.bases[qubit]
                    .computational_basis_indices.values()]

    def renormalize(self):
        """Renormalize to trace one."""
        tr = self.trace()
        if tr > 1e-8:
            self._data *= np.float(1 / tr)
        else:
            warnings.warn(
                "Density matrix trace is 0; likely your further computation "
                "will fail. Have you projected DM on a state with zero weight?")
        return tr

    def copy(self):
        """Return a deep copy of this Density."""
        data_cp = self._data.copy()
        cp = self.__class__(self.bases, data=data_cp)
        return cp

    def _cached_gpuarray(self, array):
        """
        Given a numpy array,
        calculate the python hash of its bytes;

        If it is not found in the cache, upload to gpu
        and store in cache, otherwise return cached allocation.
        """

        array = np.ascontiguousarray(array)
        key = hash(array.tobytes())
        try:
            array_gpu = self._gpuarray_cache[key]
        except KeyError:
            array_gpu = ga.to_gpu(array)
            self._gpuarray_cache[key] = array_gpu

        # for testing: read_back_and_check!

        return array_gpu

    def _check_cache(self):
        for k, v in self._gpuarray_cache.items():
            a = v.get().tobytes()
            assert hash(a) == k
