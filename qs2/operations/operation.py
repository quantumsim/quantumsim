import abc
from collections import namedtuple
from functools import reduce, lru_cache
from itertools import chain

import numpy as np
from ..bases import general
from .compiler import ChainCompiler


class Operation(metaclass=abc.ABCMeta):
    """A metaclass for all quantum operations.

    Every operation has to implement call method, that takes a
    :class:`qs2.state.State` object and modifies it inline. This method may
    return nothing or a result of a measurement, if the operation is a
    measurement.
    """

    @property
    @abc.abstractmethod
    def dim_hilbert(self):
        """Hilbert dimensionality of qubits the operation acts onto."""
        pass

    @property
    @abc.abstractmethod
    def num_qubits(self):
        """Hilbert dimensionality of qubits the operation acts onto."""
        pass

    @abc.abstractmethod
    def __call__(self, state, *qubit_indices):
        """Applies the operation inline (modifying the state) to the state
        to certain qubits. Number of qubit indices should be aligned with a
        dimensionality of the operation.
        """
        pass

    @abc.abstractmethod
    def compile(self, bases_in=None, bases_out=None):
        """Return an optimized version of this circuit, based on the
        restrictions on input and output bases, that may not be full.

        Parameters
        ----------
        bases_in : list of qs2.bases.PauliBasis or None
            Input bases of the qubits. If `None` provided, full :math:`01xy`
            basis is assumed.
        bases_out : list of qs2.bases.PauliBasis
            Output bases of the qubits. If `None` provided, full :math:`01xy`
            basis is assumed.

        Returns
        -------
        qs2.operations.Operation
            Optimized representation of self.
        """
        pass

    def at(self, *indices):
        """Returns a container with the operation, that provides also dumb
        indices of qubits it acts on. Used during processes' concatenation
        to match qubits they act onto.

        Parameters
        ----------
        i0, ..., iN: int
            Dumb indices of qubits operation acts onto.

        Returns
        -------
        _IndexedOperation
            Intermediate representation of an operation.
        """
        if not self.num_qubits == len(indices):
            raise ValueError('Number of indices is not equal to the number of '
                             'qubits in the operation.')
        return _IndexedOperation(self, indices)


_IndexedOperation = namedtuple('_IndexedOperation', ['operation', 'indices'])


class Transformation(Operation):
    """Generic transformation of a state.

    Any transformation, that should be a completely positive map, can be
    represented in a form of set of Kraus operators [1]_ or as a Pauli transfer
    matrix [2]_. Dependent on your preferred representation, you may construct
    the transformation with :func:`CompletelyPositiveMap.from_kraus` or
    :func:`CompletelyPositiveMap.from_ptm`. Constructor of this class is not
    supposed to be called in user code.

    Attributes
    ----------
    sv_cutoff : float
        During the Pauli transfer matrix optimizations, singular value
        decomposition of a transfer matrix is used to determine optimal
        computational basis. All singular values less than `sv_cutoff`
        are considered weakly contributed and neglected. This attribute
        should be set before any compilation of a circuit, otherwise default
        is used (1e-5).

    References
    ----------
    .. [1] M. A. Nielsen, I. L. Chuang, "Quantum Computation and Quantum
       Information" (Cambridge University Press, 2000).
    .. [2] D. Greenbaum, "Introduction to Quantum Gate Set Tomography",
       arXiv:1509.02921 (2000).
    """

    sv_cutoff = 1e-5

    def __init__(self, *, _i_know_what_i_do=False):
        if not _i_know_what_i_do:
            raise RuntimeError(
                'TracePreservingProcess\'s constructor is not supposed to be '
                'used explicitly. Use TracePreservingProcess.from_ptm() or '
                'TracePreservingProcess.from_kraus()')
        self._ptm = None
        self._kraus = None
        self._bases_in = None
        self._bases_out = None
        self._dim_hilbert = None
        self._num_qubits = None

    @classmethod
    def from_ptm(cls, ptm, bases_in, bases_out):
        """Construct completely positive map, based on Pauli transfer matrix.

        TODO: elaborate on PTM format.

        Parameters
        ----------
        ptm: array_like
            Pauli transfer matrix in a form of Numpy array
        bases_in: tuple of qs2.bases.PauliBasis
            Input bases of qubits.
        bases_out: tuple of qs2.bases.PauliBasis
            Output bases of qubits. If None, assumed to be the same as input
            bases.

        Returns
        -------
        Transformation
            Resulting operation
        """
        out = cls(_i_know_what_i_do=True)
        out._bases_in = bases_in
        out._bases_out = bases_out
        out._dim_hilbert = bases_in[0].dim_hilbert
        out._num_qubits = len(bases_in)
        if out._num_qubits != len(bases_out):
            raise ValueError('Number of qubits should be the same in bases_in '
                             '(has {} qubits) and bases_out (has {} qubits)'
                             .format(len(bases_in), len(bases_out)))
        for b in chain(bases_in, bases_out):
            if b.dim_hilbert != out._dim_hilbert:
                raise ValueError(
                    'All bases must have the same Hilbert dimensionality.')
        out._validate_bases(bases_out=bases_out)
        shape = tuple(b.dim_pauli for b in chain(bases_out, bases_in))
        if not ptm.shape == shape:
            raise ValueError(
                'Shape of `ptm` is not compatible with the `bases` '
                'dimensionality: \n'
                '- expected shape from provided `bases`: {}\n'
                '- `ptm` shape: {}'.format(shape, ptm.shape))
        out._ptm = ptm
        return out

    @classmethod
    def from_kraus(cls, kraus, dim_hilbert=2):
        """Construct completely positive map, based on a set of Kraus matrices.

        TODO: elaborate on Kraus matrices format.

        Parameters
        ----------
        kraus: array_like
            Pauli transfer matrix in a form of Numpy array
        dim_hilbert: int
            Dimensionality of qudits in the operation.

        Returns
        -------
        Transformation
            Resulting operation
        """
        out = cls(_i_know_what_i_do=True)
        out._dim_hilbert = dim_hilbert

        if not isinstance(kraus, np.ndarray):
            kraus = np.array(kraus)
        if len(kraus.shape) == 2:
            kraus = kraus.reshape((1, *kraus.shape))
        elif len(kraus.shape) != 3:
            raise ValueError(
                '`kraus` should be a 2D or 3D array, got shape {}'
                .format(kraus.shape))

        kraus_size = kraus.shape[1]
        if kraus_size != kraus.shape[2]:
            raise ValueError('Kraus operators should be square matrices, got '
                             '{}x{}'.format(kraus.shape[1], kraus.shape[2]))
        if kraus_size == dim_hilbert:
            out._num_qubits = 1
        elif kraus_size == dim_hilbert ** 2:
            out._num_qubits = 2
        else:
            raise ValueError(
                'Expected {n1}x{n1} (single-qubit transformation) or {n2}x{n2} '
                '(two-qubit transformation) Kraus matrices, got {n}x{n}.'
                .format(n=kraus_size, n1=dim_hilbert, n2=dim_hilbert**2))

        out._kraus = kraus
        return out

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def shape(self):
        """Shape of a PTM, that represents the operation, qubit-wise.

        For example, for a single-qubit gate, acting in a full basis,
        shape should be :math:`\\left(d^2, d^2\\right)`, where d is a Hilbert
        dimensionality of a qubit subspace. First item corresponds to the
        output dimensionality, second -- to the input one. For a two-qubit
        gate it will be :math:`\\left(d^2, d^2, d^2, d^2\\right)`.
        If PTM acts on a reduced basis or reduces a basis (for example,
        it is a projection), elements can be less than :math:`d^2`.
        """
        if self._ptm is not None:
            return self._ptm.shape
        else:
            return (self.dim_hilbert**2,) * (self.num_qubits*2)

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def size(self):
        return np.product(self.shape)

    def ptm(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in
        if (self._ptm is not None and bases_in == self._bases_in and
                bases_out == self._bases_out):
            return self._ptm
        self._validate_bases(bases_in=bases_in, bases_out=bases_out)
        shape = tuple(b.dim_pauli for b in chain (bases_out, bases_in))
        if self._kraus is not None:
            return np.einsum("xab, zbc, ycd, zad -> xy",
                             self._combine_bases_vectors(bases_out),
                             self._kraus,
                             self._combine_bases_vectors(bases_in),
                             self._kraus.conj(),
                             optimize=True).real.reshape(shape)
        if self._ptm is not None:
            return np.einsum("xij, yji, yz, zkl, wlk -> xw",
                             self._combine_bases_vectors(bases_out),
                             self._combine_bases_vectors(self._bases_out),
                             self._ptm,
                             self._combine_bases_vectors(self._bases_in),
                             self._combine_bases_vectors(bases_in),
                             optimize=True).real.reshape(shape)
        raise RuntimeError("Neither `self._kraus`, nor `self._ptm` are set.")

    def compile(self, bases_in=None, bases_out=None):
        opt_bases = self.optimal_bases(bases_in, bases_out)
        return self.from_ptm(self.ptm(*opt_bases), *opt_bases)

    def optimal_bases(self, bases_in=None, bases_out=None):
        """Based on input or output bases provided, determine an optimal
        basis, throwing away all basis elements, that are guaranteed not to
        contribute to the result of PTM application.

        Circuits provide some restrictions on input and output basis. For
        example, after the ideal initialization gate system is guaranteed to
        stay in :math:`|0\rangle` state, which means that input basis will
        consist of a single element. Similarly, if after the gate application
        qubit will be measured, only :math:`|0\rangle` and :math:`|1\rangle`
        states need to be computed, therefore we may reduce output basis to
        the classical subbasis. This method is used to perform such sort of
        optimization: usage of subbasis instead of a full basis in a density
        matrix will exponentially reduce memory consumption and computational
        time.

        Parameters
        ----------
        bases_in : tuple of qs2.bases.PauliBasis or None
            Basis of input elements, that is involved in computation. If
            None, either internal representation is taken (if available) or
            full :func:`qs2.bases.general` basis is used.
        bases_out : tuple of qs2.bases.PauliBasis or None
            Basis of output elements, that is involved in computation. If
            None, either internal representation is taken (if available) or
            full :func:`qs2.bases.general` basis is used.

        Returns
        -------
        opt_basis_in, opt_basis_out: tuple of qs2.bases.PauliBasis
            Subbases of input bases, that will contribute to computation.
        """
        bases_in = (bases_in or
                    self._bases_in or
                    (general(self.dim_hilbert),) * self.num_qubits)
        bases_out = (bases_out or
                     self._bases_out or
                     (general(self.dim_hilbert),) * self.num_qubits)
        d_in = np.prod([b.dim_pauli for b in bases_in])
        d_out = np.prod([b.dim_pauli for b in bases_out])
        u, s, vh = np.linalg.svd(self.ptm(bases_in, bases_out)
                                     .reshape(d_out, d_in),
                                 full_matrices=False)
        (truncate_index,) = (s > self.sv_cutoff).shape

        # mask_in = np.count_nonzero(vh[:truncate_index], axis=0) \
        #             .reshape(tuple(b.dim_pauli for b in bases_in)) \
        #             .nonzero()
        mask_in = np.any(np.abs(vh[:truncate_index]) > 1e-13, axis=0) \
                    .reshape(tuple(b.dim_pauli for b in bases_in)) \
                    .nonzero()
        mask_out = np.any(np.abs(u[:, :truncate_index]) > 1e-13, axis=1) \
                     .reshape(tuple(b.dim_pauli for b in bases_out)) \
                     .nonzero()

        opt_bases_in = []
        opt_bases_out = []
        for opt_bases, bases, mask in (
                (opt_bases_in, bases_in, mask_in),
                (opt_bases_out, bases_out, mask_out)):
            for basis, involved_indices in zip(bases, mask):
                # Figure out what single-qubit basis elements are not
                # involved at all
                unique_indices = np.unique(involved_indices)
                if len(unique_indices) < basis.dim_pauli:
                    # We can safely use a subbasis
                    opt_bases.append(basis.subbasis(unique_indices))
                else:
                    # Nothing can be thrown out
                    opt_bases.append(basis)

        return tuple(opt_bases_in), tuple(opt_bases_out)

    @staticmethod
    @lru_cache(maxsize=8)
    def _combine_bases_vectors(bases):
        return reduce(np.kron, [b.vectors for b in bases])

    def _validate_bases(self, **kwargs):
        for name, bases in kwargs.items():
            if not isinstance(bases, tuple):
                raise ValueError(
                    "`{n}` should be a tuple, got {t}."
                    .format(n=name, t=type(bases)))
            for b in bases:
                if self.dim_hilbert != b.dim_hilbert:
                    raise ValueError(
                        "Expected bases with Hilbert dimensionality {}, "
                        "but {} has elements with Hilbert dimensionality {}."
                        .format(self.dim_hilbert, name, b.dim_hilbert))

    def __call__(self, state, *qubit_indices):
        # FIXME state should know its basis
        proc_subspaces = self.num_qubits

        if len(qubit_indices) != proc_subspaces:
            raise ValueError(
                'Incorrect number of indicies for a single qubit PTM')

        if proc_subspaces == 1:
            state.apply_single_qubit_ptm(
                *qubit_indices,
                self.ptm((state.bases[qubit_indices[0]],)))
        elif proc_subspaces == 2:
            state.apply_two_qubit_ptm(
                *qubit_indices,
                self.ptm((state.bases[qubit_indices[0]],
                          state.bases[qubit_indices[1]])))
        else:
            raise NotImplementedError


class Initialization(Operation):

    @property
    def dim_hilbert(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def num_qubits(self):
        raise NotImplementedError

    def __call__(self, state, *qubit_indices):
        """Not implemented yet as I am unsure how the state will look.
        Should be fairly straightforward to do so (state projection)
        """
        raise NotImplementedError

    def compile(self, bases_in, bases_out):
        raise NotImplementedError


class Projection(Operation):
    @property
    def dim_hilbert(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def num_qubits(self):
        pass

    # FIXME Sampler should not be there.
    def __call__(self, state, sampler, *qubit_indices):
        """Returns the result of the measurement

        NOTE: I don't think sampler should be part of the process. However the
        measurement process needs information of the probability tree and which
        state to project. I think the bast thing is to pass tuples of
        (index, proj_state) and let the declared state be handled by the Gate
        class.
        """
        results = []
        for ind in qubit_indices:
            probs = state.partial_trace(ind)
            declared_state, proj_state, cond_prob = \
                sampler.send(probs)

            state.project(ind, proj_state)
            results.append(tuple(declared_state, proj_state, cond_prob))
        return results

    def compile(self, bases_in, bases_out):
        raise NotImplementedError


class Chain(Operation):
    """
    A chain of operations, that are applied sequentially.

    Parameters
    ----------
    op0, ..., opN: _IndexedOperation
        Operations with indices they are applied to
    """
    def __init__(self, *operations):
        self._dim_hilbert = operations[0].operation.dim_hilbert
        for op in operations[1:]:
            if op.operation.dim_hilbert != self._dim_hilbert:
                raise ValueError('All operations in the chain must have the '
                                 'same Hilbert dimensionality.')
        all_indices = np.unique(list(chain(*(op.indices for op in operations))))
        if all_indices[0] != 0 or all_indices[-1] != len(all_indices) - 1:
            raise ValueError('Indices of operations must form an ordered set '
                             'from 0 to N-1')
        self._num_qubits = len(all_indices)

        joined_ops = []
        for op_indices in operations:
            # Flatten the operations chain
            if isinstance(op_indices.operation, Chain):
                for sub_ops, sub_indices in op_indices.operation.operations:
                    op, indices = op_indices
                    new_indices = tuple((indices[i] for i in sub_indices))
                    joined_ops.append(_IndexedOperation(op, new_indices))
            else:
                joined_ops.append(op_indices)

        self.operations = joined_ops

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def num_qubits(self):
        return self._num_qubits

    def __call__(self, state, *qubit_indices):
        if len(qubit_indices) != self._num_qubits:
            raise ValueError('This is a {}-qubit operation, number of qubit '
                             'indices provided is {}'
                             .format(self._num_qubits, len(qubit_indices)))
        results = []
        for op, indices in self.operations:
            result = op(state, *(qubit_indices[i] for i in indices))
            if result is not None:
                results.append(result)
        return results if len(results) > 0 else None

    def compile(self, bases_in=None, bases_out=None):
        compiler = ChainCompiler(self)
        return compiler.compile(bases_in, bases_out)
