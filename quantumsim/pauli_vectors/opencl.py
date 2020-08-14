# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2020 Brian Tarasinski, Viacheslav Ostroukh, Boris Varbanov
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt
import os
import sys
import warnings

import numpy as np
import pyopencl as cl
import pyopencl.array as ga
import pyopencl.reduction
import pytools

from .pauli_vector import PauliVectorBase


KERNELS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'primitives.cl')
if not os.path.exists(KERNELS_FILE):
    raise ImportError("Could not find OpenCL kernels file")


class PauliVectorOpenCL(PauliVectorBase):
    """Create a new density matrix for several qudits.

    Parameters
    ----------
    bases : list of quantumsim.bases.PauliBasis
        Dimensions of qubits in the system.
    pv : array or None.
        Must be of size (2**no_qubits, 2**no_qubits). Only upper triangle
        is relevant.  If data is `None`, create a new density matrix with
        all qubits in ground state.
    force : bool, optional
        By default creation of too large density matrix (more than
        :math:`2^22` elements currently) is not allowed. Set this to `True`
        if you know what you are doing.
    context: pyopencl.Context, optional
        OpenCL context. By default, `pyopencl.create_some_context()` will
        be called to create one automatically.
    """
    dtype = np.float64
    itemsize = np.dtype(dtype).itemsize
    _gpuarray_cache = {}

    def __init__(self, bases, pv=None, *, force=False, context=None):
        super().__init__(bases, pv, force=force)

        # Context initialization
        self._context = context or cl.create_some_context()
        self._queue = cl.CommandQueue(self._context)

        # OpenCL kernels initialization
        with open(KERNELS_FILE, 'r') as f:
            kernels = cl.Program(self._context, f.read()).build()

        self._cl_multitake = kernels.multitake
        self._cl_applyptm = kernels.two_qubit_general_ptm
        self._cl_sum_along_axis = cl.reduction.ReductionKernel(
            ctx=self._context,
            dtype_out=np.float64,
            neutral="0", reduce_expr="a+b",
            map_expr="(i/stride) % dim == offset ? in[i] : 0",
            arguments="const double *in, unsigned int stride, unsigned int dim, "
                      "unsigned int offset"
        )

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

            self._data = cl.array.to_device(
                self._queue, np.ascontiguousarray(pv, dtype=self.dtype))
        elif isinstance(pv, cl.array.Array):
            if pv.dtype != self.dtype:
                raise ValueError(
                    '`pv` must have {} data type, got {}'
                    .format(self.dtype, pv.dtype)
                )
            self._data = pv
        else:
            raise ValueError(
                "`pv` must be Numpy array, PyCUDA GPU array or "
                "None, got type `{}`".format(type(pv)))

        self._work_data = cl.array.empty_like(self._data)

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

    def _ensure_gpu_array_shape(self, arr, shape):
        """
        This function tries to reuse GPU allocation of `arr`, if it is possible,
        and casts it to the shape `shape`. If required memory of `arr` allocation
        is not sufficient -- release it and allocate larger memory blob.

        Parameters
        ----------
        arr: pyopencl.array.Array
            An old array.
        shape: tuple of int
            Required output array shape

        Returns
        -------
        pyopencl.array.Array
    `   """
        new_size = pytools.product(shape)
        new_size_bytes = new_size * self.itemsize

        if self._work_data.data.size < new_size_bytes:
            # reallocate
            arr.data.release()
            self._queue.finish()  # TODO: is there any other way to force-release?
            return cl.array.empty(self._queue, shape, self.dtype)
        else:
            # reallocation not required,
            # reshape but reuse allocation
            return cl.array.Array(
                self._queue,
                shape=shape,
                dtype=self.dtype,
                data=self._work_data.data,
            )

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
        rest_shape = new_shape.copy()
        rest_shape[qubit1] = 1
        rest_shape[qubit0] = 1

        new_shape = tuple(new_shape)
        self._work_data = self._ensure_gpu_array_shape(self._work_data, new_shape)
        ptm_gpu = self._cached_gpuarray(ptm)

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

        buf = cl.LocalMemory(self.itemsize * sh_mem_size)
        self._cl_applyptm(
            self._queue,
            grid,
            block,
            self._data.data,
            self._work_data.data,
            ptm_gpu.data,
            buf,
            np.uint32(dim0_in), np.uint32(dim1_in),
            np.uint32(dim_z),
            np.uint32(dim_y),
            np.uint32(dim_rho),
            g_times_l=True,
        )

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
        if len(ptm.shape) != 2:
            raise ValueError(
                "`ptm` must be a 2D array, got {}D".format(len(ptm.shape)))

        dim_bit_out, dim_bit_in = ptm.shape
        new_shape[qubit] = dim_bit_out
        assert new_shape[qubit] == dim_bit_out
        new_size = pytools.product(new_shape)

        new_shape = tuple(new_shape)
        self._work_data = self._ensure_gpu_array_shape(self._work_data, new_shape)
        ptm_gpu = self._cached_gpuarray(ptm)

        dint = min(64, self._data.size // dim_bit_in)
        block = (1, dim_bit_out, dint)
        blocksize = dim_bit_out * dint
        grid_size = max(1, (new_size - 1) // blocksize + 1)
        grid = (grid_size, 1, 1)

        dim_z = pytools.product(self._data.shape[qubit + 1:])
        dim_y = pytools.product(self._data.shape[:qubit])
        dim_rho = new_size  # self.data.size

        buf = cl.LocalMemory(self.itemsize * (ptm.size + blocksize))
        self._cl_applyptm(
            self._queue,
            grid,
            block,
            self._data.data,
            self._work_data.data,
            ptm_gpu.data,
            buf,
            np.uint32(1), np.uint32(dim_bit_in),
            np.uint32(dim_z),
            np.uint32(dim_y),
            np.uint32(dim_rho),
            g_times_l=True,
        )

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
        diag_shape = tuple((db.dim_pauli for db in diag_bases))
        diag_size = pytools.product(diag_shape)

        if target_array is None:
            target_array = self._ensure_gpu_array_shape(self._work_data, diag_shape)
        else:
            if target_array.size < diag_size:
                raise ValueError(
                    "Size of `target_gpu_array` is too small ({}).\n"
                    "Should be at least {} ."
                    .format(target_array.size, diag_size))
            target_array = self._ensure_gpu_array_shape(target_array, diag_shape)

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
            cl.enqueue_copy(self._queue, target_array.data, self._data.data,
                            byte_count=self._data.size * self.itemsize)
        else:
            self._cl_multitake(
                self._queue,
                grid,
                block,
                self._data.data,
                target_array.data,
                idx_i_gpu.data,
                idx_j_gpu.data,
                xshape_gpu.data,
                yshape_gpu.data,
                np.uint32(len(yshape)),
                g_times_l=True,
            )

        if get_data:
            if flatten:
                return target_array.get().ravel()[:diag_size]
            else:
                return (target_array.get().ravel()[:diag_size]
                        .reshape(diag_shape))
        else:
            return target_array

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
            pt = self._cl_sum_along_axis(diag, stride, dim, offset, queue=self._queue)
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
        raise NotImplementedError

    def copy(self):
        """Return a deep copy of this Density."""
        raise NotImplementedError

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
            array_gpu = cl.array.to_device(self._queue, array)
            self._gpuarray_cache[key] = array_gpu

        return array_gpu