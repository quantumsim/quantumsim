import abc
from functools import reduce
import numpy as np
from .operators import Operator, PTMOperator
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
    ... rotY90 = qs2.processes.rotate_z(0.5*numpy.pi)
    ... rotY180 = qs2.processes.join(rotY90, rotY90)

    If the dimensionality of processes does not match, we may specify qubits it
    acts onto in a form of integer dumb indices:

    >>> cz = qs2.processes.cphase()
    ... cnot = qs2.processes.join(rotY90.at(0), cz.at(0, 1), rotY90.at(0))

    This is also useful, if you want to combine processes on different
    qubits into one:

    >>> hadamard = qs2.processes.hadamard()
    ... hadamard3q = qs2.processes.join(hadamard.at(0), hadamard.at(1),
    ...                                  hadamard.at(2))

    All the dumb indices involved in `join` function must form an ordered set
    `0, 1, ..., N`.

    Parameters
    ----------
    indices: None or tuple
        Indices of qubits it acts on. They are designed to be dumb and used
        for tracking the multiplication of several processes.
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
        self.inds = indices


class TracePreservingProcess(Process):
    def __init__(self, operator):
        """
        NOTE: I changed the process class to represent both qubit operators and less conventional processes (measurement, initialization aka processes which are not represented by an operator). The operator based processes are now this.
        """

        if not isinstance(operator, Operator):
            raise ValueError(
                "Please provide a valid operator to define the process")
        self.operator = operator

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
        return _IndexedProcess(self.operator, indices)

    def prepare(self, bases_in, bases_out=None):
        self.operator = self.operator.to_ptm(bases_in, bases_out)

    def __call__(self, state, *qubit_indices):
        if not isinstance(self.operator, PTMOperator):
            raise ValueError("Cannot apply a non-PTM operator to the state")

        proc_subspaces = self.operator.num_subspaces

        if len(qubit_indices) != proc_subspaces:
            raise ValueError(
                'Incorrect number of indicies for a single qubit PTM')

        if proc_subspaces == 1:
            state.apply_single_qubit_ptm(*qubit_indices, self.operator.matrix)
        elif proc_subspaces == 2:
            state.apply_two_qubit_ptm(*qubit_indices, self.operator.matrix)
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

        #NOTE: I don't think sampler should be part of the process. However the measurement process needs information of the probability tree and which state to project. I think the bast thing is to pass tuples of (index, proj_state) and let the declared state be handled by the Gate class.
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
        combined_oper = PTMOperator(combined_ptm, full_bases)
        return TracePreservingProcess(combined_oper)

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
    result_ndims = 2 * num_subspaces
    result_inds = [i for i in range(result_ndims)]

    full_bases = [general(dim_hilbert)
                  for dim_hilbert in subspaces_dim_hilbert.values()]
    pauli_dims = [basis.dim_pauli for basis in full_bases]
    op_dim_pauli = np.prod(pauli_dims)
    combined_ptm = np.eye(op_dim_pauli).reshape(pauli_dims + pauli_dims)

    for indexed_process in processes:
        proc_inds = indexed_process.inds
        num_inds = len(proc_inds)
        if num_inds == 1:
            index = proc_inds[0]
            conv_op = indexed_process.op.to_ptm((full_bases[index],))
            _op_inds = [index, result_ndims]
            _ptm_inds = result_inds.copy()
            _ptm_inds[index] = result_ndims
            combined_ptm = np.einsum(
                conv_op.matrix, _op_inds, combined_ptm, _ptm_inds, result_inds, optimize=True)

        elif num_inds == 2:
            end_inds = [ind + result_ndims for ind in proc_inds]
            conv_basis = tuple(full_bases[ind] for ind in proc_inds)
            conv_op = indexed_process.op.to_ptm(conv_basis)
            _dim_paulis = [basis.dim_pauli for basis in conv_basis]
            conv_ptm = conv_op.matrix.reshape(_dim_paulis + _dim_paulis)
            _op_inds = list(proc_inds) + end_inds
            _ptm_inds = end_inds + result_inds[num_subspaces:]
            combined_ptm = np.einsum(
                conv_ptm, _op_inds, combined_ptm, _ptm_inds, optimize=True)
        else:
            raise NotImplementedError

    combined_ptm = combined_ptm.reshape(op_dim_pauli, op_dim_pauli)
    combined_oper = PTMOperator(combined_ptm, tuple(full_bases))
    return TracePreservingProcess(combined_oper)


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
    linear_oper = PTMOperator(linear_ptm, full_bases)
    return TracePreservingProcess(linear_oper)
