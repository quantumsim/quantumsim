# This file is part of quantumsim. (https://github.com/brianzi/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt
import sys

import numpy as np
import os
import pytools
import warnings
import pyopencl as cl
import pyopencl.array as ga

from .pauli_vector import PauliVectorBase


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
    _gpuarray_cache = {}

    def __init__(self, bases, pv=None, *, force=False, context=None):
        super().__init__(bases, pv, force=force)
        self._context = context or cl.create_some_context()
        self._queue = cl.CommandQueue(self._context)

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

            self._data = cl.array.to_device(self._queue, pv.astype(np.float64))
        elif isinstance(pv, cl.array.Array):
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
        raise NotImplementedError

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
        self._validate_qubit(qubit, 'bit')
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError