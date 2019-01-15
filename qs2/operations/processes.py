import abc
from functools import reduce, lru_cache
import numpy as np
from ..bases import general


class Process(metaclass=abc.ABCMeta):
    """A metaclass for all processes.

    Every operation has to implement call method, that takes a
    :class:`qs2.state.State` object and modifies it inline. This method may
    return nothing or a result of a measurement, if the operation is a
    measurement.

    Processs are designed to form an algebra. For the sake of making
    interface sane, we do not provide explicit multiplication, instead we use
    :func:`qs2.processes.join` function to concatenate a set of processes.
    I.e., if we have `rotate90` operation, we may construct `rotate180`
    operation as follows:

    >>> import qs2, numpy
    ... rotY90 = qs2.operations.rotate_z(0.5*numpy.pi)
    ... rotY180 = qs2.operations.join(rotY90, rotY90)

    If the dimensionality of processes does not match, we may specify qubits it
    acts onto in a form of integer dumb indices:

    >>> cz = qs2.operations.cphase()
    ... cnot = qs2.operations.join(rotY90.at(0), cz.at(0, 1), rotY90.at(0))

    This is also useful, if you want to combine processes on different
    qubits into one:

    >>> hadamard = qs2.operations.hadamard()
    ... hadamard3q = qs2.operations.join(hadamard.at(0), hadamard.at(1),
    ...                                  hadamard.at(2))

    All the dumb indices involved in `join` function must form an ordered set
    `0, 1, ..., N`.
    """

    @abc.abstractmethod
    def __call__(self, state, *qubit_indices):
        """Applies the operation inline (modifying the state) to the state
        to certain qubits. Number of qubit indices should be aligned with a
        dimensionality of the operation.
        """
        pass


class _IndexedProcess:
    """Internal representation of an processes during their multiplications.
    Contains an operation itself and dumb indices of qubits it acts on.
    """
    def __init__(self, operator, indices):
        self.op = operator
        self.indices = indices


class TracePreservingProcess(Process):
    def __init__(self, *, _i_know_what_i_do=False):
        """
        NOTE: I changed the process class to represent both qubit operators and less conventional processes (measurement, initialization aka processes which are not represented by an operator). The operator based processes are now this.
        """
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
    def from_ptm(cls, ptm, bases_in, bases_out=None):
        out = cls(_i_know_what_i_do=True)
        out._bases_in = bases_in
        out._dim_hilbert = tuple([basis.dim_hilbert for basis in bases_in])
        if bases_out is not None:
            out._validate_bases(bases_out=bases_out)
            out._bases_out = bases_out
        else:
            out._bases_out = bases_in
        expected_shape = (np.prod(out.dim_pauli),) * 2
        if not ptm.shape == expected_shape:
            raise ValueError(
                'Shape of `ptm` is not compatible with the `bases` '
                'dimensionality: \n'
                '- expected shape from provided `bases`: {}\n'
                '- `ptm` shape: {}'.format(expected_shape, out._ptm.shape))
        out._ptm = ptm
        return out

    @classmethod
    def from_kraus(cls, kraus, dim_hilbert):
        out = cls(_i_know_what_i_do=True)
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
        out._dim_hilbert = tuple(dim_hilbert)
        return out

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def dim_pauli(self):
        return tuple((d * d for d in self._dim_hilbert))

    @property
    def num_subspaces(self):
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
        _IndexedProcess
            Intermediate representation of an operation.
        """
        return _IndexedProcess(self, indices)

    @lru_cache(maxsize=32)
    def ptm(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in
        if (self._ptm is not None and bases_in == self._bases_in and
                bases_out == self._bases_out):
            return self._ptm
        self._validate_bases(bases_in=bases_in, bases_out=bases_out)
        if self._kraus is not None:
            return np.einsum("xab, zbc, ycd, zad -> xy",
                             self._combine_bases_vectors(bases_out),
                             self._kraus,
                             self._combine_bases_vectors(bases_in),
                             self._kraus.conj(),
                             optimize=True).real
        if self._ptm is not None:
            return np.einsum("xij, yji, yz, zkl, wlk -> xw",
                             self._combine_bases_vectors(bases_out),
                             self._combine_bases_vectors(self._bases_out),
                             self._ptm,
                             self._combine_bases_vectors(self._bases_in),
                             self._combine_bases_vectors(bases_in),
                             optimize=True).real
        raise RuntimeError("Neither `self._kraus`, nor `self._ptm` are set.")

    @staticmethod
    @lru_cache(maxsize=8)
    def _combine_bases_vectors(bases):
        return reduce(np.kron, [basis.vectors for basis in bases])

    def _validate_bases(self, **kwargs):
        for name, bases in kwargs.items():
            dim_hilbert = tuple((b.dim_hilbert for b in bases))
            if self.dim_hilbert != dim_hilbert:
                raise ValueError(
                    "The dimensions of `{n}` do not match the operation's "
                    "Hilbert dimensionality:\n"
                    "- expected Hilbert dimensionality: {d_exp}\n"
                    "- `{n}`' Hilbert dimensionality: {d_basis}"
                    .format(n=name, d_exp=self.dim_hilbert, d_basis=dim_hilbert)
                )

    def __call__(self, state, *qubit_indices):
        # FIXME state should know its basis
        proc_subspaces = self.num_subspaces

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


class Initialization(Process):
    def __call__(self, state, *qubit_indices):
        """Not implemented yet as I am unsure how the state will look. Should be fairly straightforward to do so (state projection)
        """
        pass


class Reset(Process):
    def __call__(self, state, *qubit_indices):
        """Not implemented yet as I am unsure how the state will look. Should be fairly straightforward to do so (state projection)
        """

        pass


class Measurement(Process):
    def __call__(self, state, sampler, *qubit_indices):
        """Returns the result of the measurement

        #NOTE: I don't think sampler should be part of the process.
         However the measurement process needs information of the probability tree and which state to project. I think the bast thing is to pass tuples of (index, proj_state) and let the declared state be handled by the Gate class.
        """
        results = []
        for ind in qubit_indices:
            probs = state.partial_trace(ind)
            declared_state, proj_state, cond_prob = \
                sampler.send(probs)

            state.project(ind, proj_state)
            results.append(tuple(declared_state, proj_state, cond_prob))
        return results


def join(*processes):
    """Combines a list of processes into one operation.

    Parameters
    ----------
    op0, ..., opN: qs2.Process or qs2.operation._DumbIndexedProcess
        Processs involved, in chronological order. If number of qubits in
        them is not the same everywhere, all of them must specify dumb
        indices of qubits they act onto with `Process.at()` method.

    #TODO: Products formed in the general basis by first converting operators to ptm. This might be not efficient, but makes the function easier and more general. We might want to optionally extend this to be specifiable by the use (or something smarter). This might not be too bad as this allows some sparsity analysis to be performed and for basis optimization.

    #TODO: Currently only joins up to two-subspace operators. This is ok for now as we still can't apply three or more subspace operators to the state. However it'd be good to generalize and make this as agnostic of dimensions/subspaces as possible for the futre.

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
    if isinstance(init_proc, TracePreservingProcess):
        cls = TracePreservingProcess
    elif isinstance(init_proc, _IndexedProcess):
        cls = _IndexedProcess
    else:
        raise ValueError(
            'Expected a process, got {}'.format(type(init_proc)))

    if not all(isinstance(proc_inst, cls) for proc_inst in processes[1:]):
        raise ValueError(
            'Specify indices for all processes involved with '
            '`Process.at()` method.')

    if cls is TracePreservingProcess:
        req_num_subspaces = init_proc.operator.num_subspaces

        if not all(proc.operator.num_subspaces == req_num_subspaces
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
        return TracePreservingProcess.from_ptm(combined_ptm, full_bases)

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
    return TracePreservingProcess.from_ptm(combined_ptm, full_bases)


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

    if not all(isinstance(proc_inst, TracePreservingProcess)
               for proc_inst in processes):
        raise ValueError('All processes need to be Trace Perserving')

    req_num_subspaces = processes[0].operator.num_subspaces

    if not all(proc.operator.num_subspaces == req_num_subspaces
               for proc in processes):
        raise ValueError(
            'If joining processes the number of subspaces of each process must be the same.')
    req_dim_hilbert = processes[0].operator.dim_hilbert
    full_bases = tuple(general(dim_hilbert)
                       for dim_hilbert in req_dim_hilbert)

    conv_ptms = [process.operator.to_ptm(
        full_bases).matrix for process in processes]
    linear_ptm = [weight * ptm for weight, ptm in zip(weights, conv_ptms)]
    return TracePreservingProcess.from_ptm(linear_ptm, full_bases)
