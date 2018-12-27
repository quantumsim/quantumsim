import abc
from .operators import Operator, PTMOperator


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
    def prepare(self, bases=None):
        """Prepares a proccess so that it the operater implemented the specific process can be applied to a state.
        """

        pass

    @abc.abstractmethod
    def __call__(self, state, *qubit_indices):
        """Applies the operation inline (modifying the state) to the state
        to certain qubits. Number of qubit indices should be aligned with a
        dimensionality of the operation.
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
        _DumbIndexedProcess
            Intermediate representation of an operation.
        """
        return _DumbIndexedProcess(self, indices)


class _DumbIndexedProcess:
    """Internal representation of an processes during their multiplications.
    Contains an operation itself and dumb indices of qubits it acts on.
    """

    def __init__(self, operation, indices):
        self._operation = operation
        self._indices = indices


class TracePreservingProcess(Process):
    """A general trace preserving operation.

    Parameters
    ----------
    transfer_matrix: array_like, optional
        Pauli transfer matrix of the operation.
    kraus: list of array_like, optional
        Kraus representation of the operation.
    basis: qs2.basis.PauliBasis
        Basis, in which the operation is provided.
        TODO: expand.
    """

    def __init__(self, operator):
        if not isinstance(operator, Operator):
            raise ValueError(
                "Please provide a valid operator to define the process")
        self.operator = operator

    def prepare(self, bases=None):
        self.operator = self.operator.to_ptm(bases)

    def __call__(self, state, *qubit_indices):
        if not isinstance(self.operator, PTMOperator):
            raise ValueError("Cannot apply a non-PTM operator to the state")

        op_subspaces = self.operator.num_subspaces

        if len(qubit_indices) != op_subspaces:
            raise ValueError(
                'Incorrect number of indicies for a single qubit PTM')

        if op_subspaces == 1:
            state.apply_single_qubit_ptm(*qubit_indices, self.operator.matrix)
        elif op_subspaces == 2:
            state.apply_two_qubit_ptm(*qubit_indices, self.operator.matrix)
        else:
            raise NotImplementedError


class Initialization(Process):
    def __call__(self, state, *qubit_indices):
        pass

    @property
    def num_qubits(self):
        raise NotImplementedError()


class Measurement(Process):
    def __call__(self, state, *qubit_indices):
        """Returns the result of the measurement"""
        pass

    @property
    def num_qubits(self):
        raise NotImplementedError()


class CombinedProcess(Process):
    def __init__(self, operators, joint_bases):
        pass

    def __call__(self, state, *qubit_indices):
        pass

    def prepare(self, bases=None):
        pass


def join(*processes, out_bases=None):
    """Combines a list of processes into one operation.

    Parameters
    ----------
    op0, ..., opN: qs2.Process or qs2.operation._DumbIndexedProcess
        Processs involved, in chronological order. If number of qubits in
        them is not the same everywhere, all of them must specify dumb
        indices of qubits they act onto with `Process.at()` method.

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
    if isinstance(init_proc, Process):
        cls = Process
    elif isinstance(init_proc, _DumbIndexedProcess):
        cls = _DumbIndexedProcess
    else:
        raise ValueError(
            'Expected an operation, got {}'.format(type(init_proc)))

    if not all(isinstance(proc_inst, cls) for proc_inst in processes[1:]):
        raise ValueError(
            'Specify indices for all processes involved with '
            '`Process.at()` method.')

    raise NotImplementedError()
    # for each operation: get basis, check that the basis match and if not convert them to the same base (specified by the basis of the first operation).
