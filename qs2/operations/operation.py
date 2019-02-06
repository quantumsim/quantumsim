import abc
from functools import reduce, lru_cache
from itertools import chain

import numpy as np
from ..bases import general


class Operation(metaclass=abc.ABCMeta):
    """A metaclass for all quantum operations.

    Every operation has to implement call method, that takes a
    :class:`qs2.state.State` object and modifies it inline. This method may
    return nothing or a result of a measurement, if the operation is a
    measurement.

    Operations are designed to form an algebra. For the sake of making
    interface sane, we do not provide explicit multiplication, instead we use
    :func:`qs2.operations.join` function to concatenate a set of operations.
    I.e., if we have `rotate90` operation, we may construct `rotate180`
    operation as follows:

    >>> import qs2, numpy
    ... rotY90 = qs2.operations.rotate_z(0.5*numpy.pi)
    ... rotY180 = qs2.operations.join(rotY90, rotY90)

    If the dimensionality of operations does not match, we may specify qubits it
    acts onto in a form of integer dumb indices:

    >>> cz = qs2.operations.cphase()
    ... cnot = qs2.operations.join(rotY90.at(0), cz.at(0, 1), rotY90.at(0))

    This is also useful, if you want to combine operations on different
    qubits into one:

    >>> hadamard = qs2.operations.hadamard()
    ... hadamard3q = qs2.operations.join(hadamard.at(0), hadamard.at(1),
    ...                                  hadamard.at(2))

    All the dumb indices involved in `join` function must form an ordered set
    `0, 1, ..., N`.
    """

    @property
    @abc.abstractmethod
    def dim_hilbert(self):
        """Hilbert dimensionality of qubits the operation acts onto."""
        pass

    @property
    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def __call__(self, state, *qubit_indices):
        """Applies the operation inline (modifying the state) to the state
        to certain qubits. Number of qubit indices should be aligned with a
        dimensionality of the operation.
        """
        pass

    @abc.abstractmethod
    def compile(self, basis_in=None, basis_out=None):
        """Return an optimized version of this circuit, based on the
        restrictions on input and output bases, that may not be full.

        Parameters
        ----------
        basis_in : list of qs2.bases.PauliBasis or None
            Input bases of the qubits. If `None` provided, full :math:`01xy`
            basis is assumed.
        basis_out : list of qs2.bases.PauliBasis
            Output bases of the qubits. If `None` provided, full :math:`01xy`
            basis is assumed.

        Returns
        -------
        qs2.operations.Operation
            Optimized representation of self.
        """
        pass


class _IndexedOperation:
    """Internal representation of an operations during their multiplications.
    Contains an operation itself and dumb indices of qubits it acts on.
    """
    def __init__(self, operation, indices):
        self.op = operation
        self.indices = indices

    def __matmul__(self, other):
        """Combines an operation with another operation in a clever manner."""
        if not isinstance(other, _IndexedOperation):
            raise ValueError('RHS must be an operation')
        involved_indices = set(sorted(self.indices + other.indices))
        raise NotImplementedError


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
        out._dim_hilbert = tuple([b.dim_hilbert for b in bases_in])
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
    def from_kraus(cls, kraus, dim_hilbert):
        """Construct completely positive map, based on a set of Kraus matrices.

        TODO: elaborate on Kraus matrices format.

        Parameters
        ----------
        kraus: array_like
            Pauli transfer matrix in a form of Numpy array
        dim_hilbert: tuple of int
            Dimensionalities of qubit subspaces.

        Returns
        -------
        Transformation
            Resulting operation
        """
        out = cls(_i_know_what_i_do=True)
        out._dim_hilbert = dim_hilbert
        if not isinstance(dim_hilbert, tuple):
            raise ValueError('`dim_hilbert` should be a tuple, '
                             'got {}'.format(type(dim_hilbert)))
        if not isinstance(kraus, np.ndarray):
            kraus = np.array(kraus)
        if len(kraus.shape) == 2:
            kraus = kraus.reshape((1, *kraus.shape))
        elif len(kraus.shape) != 3:
            raise ValueError(
                '`kraus` should be a 2D or 3D array, got shape {}'
                    .format(kraus.shape))
        expected_size = np.prod(dim_hilbert)
        expected_shape = (expected_size, expected_size)
        if kraus.shape[1:] != expected_shape:
            raise ValueError(
                'Shape of `kraus` is not compatible with the `dim_hilbert`\n'
                '- expected shape from provided `dim_hilbert`: (?, {m}, {m})\n'
                '- `ptm` shape: {shape}'
                    .format(m=expected_size, shape=kraus.shape))
        out._kraus = kraus
        return out

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def shape(self):
        if self._ptm is not None:
            return self._ptm.shape
        else:
            return tuple((d * d for d in self._dim_hilbert)) * 2

    @property
    def num_qubits(self):
        return len(self.dim_hilbert)

    @property
    def size(self):
        return np.product(self.dim_pauli)

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
        return _IndexedOperation(self, indices)

    def ptm(self, bases_in, bases_out=None):
        """Return a Pauli transfer matrix in the specified input and output
        bases.

        If the Hilbert dimensionality of the bases provided is more than
        Hilbert dimensionality of `self` (say, the operation is created for
        qubits, and is requested in qutrit bases), it is assumed that it acts
        trivially on the extra subspaces. Conversion down is not allowed.

        Parameters
        ----------
        bases_in : tuple of qs2.bases.PauliBasis
            Input bases, in which PTM will be expressed.
        bases_out : tuple of qs2.bases.PauliBasis or None
            Output bases, in which PTM will be expressed. If None, is taken
            equal to the input bases.

        Returns
        -------
        array
            Pauli transfer matrix of the operation.
            TODO: refer to format
        """
        if bases_out is None:
            bases_out = bases_in
        self._validate_bases(bases_in=bases_in, bases_out=bases_out)
        needs_rescale = False
        for i, (d0, d_in, d_out) in enumerate(
                zip(self.dim_hilbert, (b.dim_hilbert for b in bases_in),
                    (b.dim_hilbert for b in bases_out))):
            if d_in != d_out:
                raise ValueError(
                    '`bases_in` and `bases_out` must have the same Hilbert '
                    'dimensionality, got {} and {} correspondingly for '
                    'their item number {}.'.format(d_in, d_out, i))
            if d0 > d_in:
                raise ValueError('Hilbert dimensionality of qubit {} in the '
                                 'basis ({}) is less than Hilbert '
                                 'dimensionality of the correspondent qubit '
                                 'in the operation ({}).'
                                 .format(i, d_in, d0))
            if d0 < d_in:
                needs_rescale = True

        if (self._ptm is not None and bases_in == self._bases_in and
                bases_out == self._bases_out):
            return self._ptm

        shape = tuple(b.dim_pauli for b in chain(bases_out, bases_in))
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
                    tuple(general(d) for d in self.dim_hilbert))
        bases_out = (bases_out or
                     self._bases_out or
                     tuple(general(d) for d in self.dim_hilbert))
        d_in = np.prod([b.dim_pauli for b in bases_in])
        d_out = np.prod([b.dim_pauli for b in bases_out])
        u, s, vh = np.linalg.svd(self.ptm(bases_in, bases_out)
                                     .reshape(d_out, d_in),
                                 full_matrices=False)
        (truncate_index,) = (s > self.sv_cutoff).shape

        mask_in = np.any(np.abs(vh[:truncate_index]) > 1e-10, axis=0) \
                    .reshape(tuple(b.dim_pauli for b in bases_in)) \
                    .nonzero()
        mask_out = np.any(np.abs(u[:, :truncate_index]) > 1e-10, axis=1) \
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
            if self.num_qubits != len(bases):
                raise ValueError(
                    '`{}` contains {} qubit bases, but the operation is '
                    '{}-qubit'.format(name, len(bases), self.num_qubits))

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

    def __call__(self, state, *qubit_indices):
        """Not implemented yet as I am unsure how the state will look.
        Should be fairly straightforward to do so (state projection)
        """
        pass

    def compile(self, basis_in, basis_out):
        pass


class Projection(Operation):
    @property
    def dim_hilbert(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

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

    def compile(self, basis_in, basis_out):
        pass


def join(*processes):
    """Combines a list of processes into one operation.

    Parameters
    ----------
    op0, ..., opN: qs2.Process or qs2.operation._DumbIndexedProcess
        Processs involved, in chronological order. If number of qubits in
        them is not the same everywhere, all of them must specify dumb
        indices of qubits they act onto with `Process.at()` method.

    TODO: Products formed in the general basis by first converting operators to
    ptm. This might be not efficient, but makes the function easier and more
    general. We might want to optionally extend this to be specifiable by the
    use (or something smarter). This might not be too bad as this allows some
    sparsity analysis to be performed and for basis optimization.

    #TODO: Currently only joins up to two-subspace operators. This is ok for
    now as we still can't apply three or more subspace operators to the state.
    However it'd be good to generalize and make this as agnostic of
    dimensions/subspaces as possible for the futre.

    Returns
    -------
    qs2.Process
        Combined operation. Exact type of the output operation may depend on
        input.
    """
    # input validation
    if len(processes) < 2:
        raise ValueError('Specify at least two processes to join.')

    init_proc = processes[0]
    if isinstance(init_proc, Transformation):
        cls = Transformation
    elif isinstance(init_proc, _IndexedOperation):
        cls = _IndexedOperation
    else:
        raise ValueError(
            'Expected a process, got {}'.format(type(init_proc)))

    if not all(isinstance(proc_inst, cls) for proc_inst in processes[1:]):
        raise ValueError(
            'Specify indices for all processes involved with '
            '`Process.at()` method.')

    if cls is Transformation:
        req_num_subspaces = init_proc.operator.num_qubits

        if not all(proc.operator.num_qubits == req_num_subspaces
                   for proc in processes[1:]):
            raise ValueError(
                'If joining processes the number of subspaces of each process must be the same.')

        # Currently joining the processes right away by doing the product internally in the full basis. Perhaps should return a class with list of operators that compiles them along the prepare function?

        req_dim_hilbert = init_proc.operator.dim_hilbert
        full_bases = tuple(general(dim_hilbert)
                           for dim_hilbert in req_dim_hilbert)

        conv_ptms = [process.operator.to_ptm(
            full_bases).matrix for process in processes]
        combined_ptm = reduce(np.dot, conv_ptms)
        return Transformation.from_ptm(combined_ptm, full_bases)

    temp_dims = {}
    for process in processes:
        for ind, dim in zip(process.inds, process.op.dim_hilbert):
            if ind not in temp_dims:
                temp_dims[ind] = dim
            else:
                if temp_dims[ind] != dim:
                    raise ValueError('Hilbert dim mismatch')

    subspaces_dim_hilbert = dict(sorted(temp_dims.items()))

    num_subspaces = len(subspaces_dim_hilbert)

    ptm_inds_1 = [n for n in range(0, num_subspaces)]
    ptm_inds_2 = [n for n in range(num_subspaces, 2 * num_subspaces)]
    ptm_inds_3 = [n for n in range(2 * num_subspaces, 3 * num_subspaces)]

    full_bases = tuple(general(dim_hilbert)
                       for dim_hilbert in subspaces_dim_hilbert.values())
    pauli_dims = [basis.dim_pauli for basis in full_bases]
    op_dim_pauli = np.prod(pauli_dims)
    combined_ptm = np.eye(op_dim_pauli).reshape(pauli_dims + pauli_dims)

    for indexed_process in processes:
        proc_inds = list(reversed(indexed_process.inds))

        conv_basis = tuple(full_bases[ind] for ind in proc_inds)
        conv_op = indexed_process.op.to_ptm(conv_basis)
        op_pauli_dim = [pauli_dims[ind] for ind in proc_inds]
        conv_ptm = conv_op.matrix.reshape(op_pauli_dim + op_pauli_dim)

        conv_ptm_inds = [ptm_inds_1[ind] for ind in proc_inds] + \
                        [ptm_inds_3[ind] for ind in proc_inds]

        reduced_prod_inds = [ptm_inds_3[i] if i in proc_inds else ptm_inds_1[i]
                             for i in range(num_subspaces)]

        combined_ptm_inds = reduced_prod_inds + ptm_inds_2
        combined_ptm = np.einsum(
            conv_ptm, conv_ptm_inds,
            combined_ptm, combined_ptm_inds,
            optimize=True)

    combined_ptm = combined_ptm.reshape(op_dim_pauli, op_dim_pauli)
    return Transformation.from_ptm(combined_ptm, full_bases)


def _linear_addition(*weighted_processes):
    """An internal function which returns a linear combination of two processes (after converting them to ptms in the general basis as an intermediate step towards finds the product). This function can be useful when two different process happen with probabilities.

    NOTE: This operation might be useful in the future (For instance if one wants to apply two difference processes each happened with a probability p_i). For now I am not using it and it seemed a bit out of place, so I have left it as an internal method.

    TODO: Current addition performed in intermediate basis. This can be changed (same as join())

    Parameters
    ----------
    (w1, op0), ..., (wN, opN): tuples of weight coefficient and  qs2.TracePersrvingProcess
        Processs involved, in chronological order. The number of subspaces involved in each process mst be the same.

    Returns
    -------
    TracePersrvingProcess
        The resulting process obtained from combined
    """

    if len(weighted_processes) < 2:
        raise ValueError('Specify at least two processes to join.')

    weights, processes = zip(*weighted_processes)

    if np.sum(weights) != 1:
        raise ValueError('The sum of the coefficients must add up to 1')

    if not all(isinstance(proc_inst, Transformation)
               for proc_inst in processes):
        raise ValueError('All processes need to be Trace Perserving')

    req_num_subspaces = processes[0].operator.num_qubits

    if not all(proc.operator.num_qubits == req_num_subspaces
               for proc in processes):
        raise ValueError(
            'If joining processes the number of subspaces of each process must be the same.')
    req_dim_hilbert = processes[0].operator.dim_hilbert
    full_bases = tuple(general(dim_hilbert)
                       for dim_hilbert in req_dim_hilbert)

    conv_ptms = [process.operator.to_ptm(
        full_bases).matrix for process in processes]
    linear_ptm = [weight * ptm for weight, ptm in zip(weights, conv_ptms)]
    return Transformation.from_ptm(linear_ptm, full_bases)
