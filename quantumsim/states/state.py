import abc
import numpy as np
from functools import reduce

from quantumsim.algebra import pv_to_dm, sigma, dm_to_pv
from quantumsim.bases import PauliBasis, general


def prod(iterable):
    # From Python 3.8 this function can be replaced to math.prod
    from operator import mul

    return reduce(mul, iterable, 1)


class State(metaclass=abc.ABCMeta):
    """A metaclass, that defines standard interface for Quantumsim density
    matrix backend.

    Every instance, that implements the interface of this class, should call
    `super().__init__` in the beginning of its execution, because a lot of
    sanity checks are done here.

    Parameters
    ----------
    qubits: list of hashable or int
        Tags of the qubits in the state. If integer is provided,
        list of qubits is initialized as `list(range(qubits))`.
    pv: array-like, optional
        Pauli vector, that represents the density matrix in the selected
        bases. If `None`, density matrix is initialized in
        :math:`\\left| 0 \\cdots 0 \\right\\rangle` state.
    bases: list of quantumsim.bases.PauliBasis, optional
        A description of the basis for the qubits. Required if `pv` is provided,
        otherwise must be left empty. If not provided, defaults to the subbasis of all
        qubits in :math:`\\left| 0 \\cdots 0 \\right\\rangle` state.
    dim_hilbert: int, optional
        Hilbert dimensionality of qubits in the state. If `bases` are provided, has no
        effect.
    force: bool
        By default creation of too large density matrix (more than
        :math:`2^22` elements currently) is not allowed. Set this to `True`
        if you know what you are doing.
    """

    _size_max = 2**22

    @abc.abstractmethod
    def __init__(self, qubits, pv=None, bases=None, *, dim_hilbert=2, force=False):
        if isinstance(qubits, int):
            self.qubits = list(range(qubits))
        else:
            self.qubits = list(qubits)
        if (pv is None) ^ (bases is None):
            raise ValueError("Both `pv` and `bases` must be provided simultaneously.")
        if bases is not None:
            self.bases = list(bases)
            self.dim_hilbert = self.bases[0].dim_hilbert
            if not all(basis.dim_hilbert == self.dim_hilbert for basis in self.bases):
                raise ValueError(
                    "All basis elements must have the same Hilbert " "dimensionality"
                )
        else:
            self.dim_hilbert = dim_hilbert
            self.bases = [general(self.dim_hilbert).subbasis([0])] * len(self.qubits)
        if self.size > self._size_max and not force:
            raise ValueError(
                "Density matrix of the system is going to have {} items. It "
                "is probably too much. If you know what you are doing, "
                "pass `force=True` argument to the constructor."
            )
        # Pauli vector storage must be initialized in the derived class

    @classmethod
    def from_pv(cls, pv, bases, qubits=None, *, force=False):
        """Construct a new State instance from existing data in a form of expansion of
        the density matrix in a basis of (generalized) Pauli matrices.

        Parameters
        ----------
        pv: array-like
            Pauli vector in a form of a numpy array
        bases: list of quantumsim.PauliBasis
            Basis for the `pv`.
        qubits: list of hashable, optional
            Tags of the qubits in the state. If not provided, defaults to
            `list(range(len(bases)))`
        force: bool, optional
            By default creation of too large density matrix (more than
            :math:`2^22` elements currently) is not allowed. Set this to `True`
            if you know what you are doing.

        Returns
        -------
        State
        """
        bases = list(bases)
        if not qubits:
            qubits = len(bases)
        return cls(qubits, pv, bases, force=force)

    @abc.abstractmethod
    def to_pv(self):
        """Get data in a form of Numpy array

        Returns
        -------
        array
        """
        pass

    @classmethod
    def from_dm(cls, dm, bases, qubits=None, *, force=False):
        """Construct a new State instance from a density matrix.

        Parameters
        ----------
        dm: array-like
            Density matrix
        bases: quantumsim.PauliBasis or list of quantumsim.PauliBasis
            Basis for storing the state.
        qubits: list of hashable, optional
            Tags of the qubits in the state. If not provided, defaults to the range
            from 0 to number of qubits minus 1.
        force: bool, optional
            By default creation of too large density matrix (more than
            :math:`2^22` elements currently) is not allowed. Set this to `True`
            if you know what you are doing.

        Returns
        -------
        State
        """
        if isinstance(bases, PauliBasis):
            n_qubits = len(dm) // bases.dim_hilbert
            bases = [bases] * n_qubits
        if not qubits:
            qubits = len(bases)
        return cls(qubits, dm_to_pv(dm, bases), bases, force=force)

    def to_dm(self):
        """
        Return data in a form of density matrix.

        Returns
        -------
        array
        """
        return pv_to_dm(self.to_pv(), self.bases)

    @property
    def dim_pauli(self):
        """(tuple of int) Number of basis elements for each of the qubit."""
        return tuple(pb.dim_pauli for pb in self.bases)

    @property
    def size(self):
        """(int) Size (number of floating point numbers in it) of the data array."""
        return prod(self.dim_pauli)

    @abc.abstractmethod
    def apply_ptm(self, ptm, *qubits):
        """Applies a Pauli transfer matrix (PTM) to this state inline.

        Parameters
        ----------
        ptm: array
            A PTM, expanded in the current basis of the state
        q1, ..., qN: hashable
            Qubits to apply PTM to
        """
        self._validate_qubits(qubits)
        if len(ptm.shape) != 2 * len(qubits):
            raise ValueError(
                f"{len(qubits)}-qubit PTM must have {2*len(qubits)} "
                f"dimensions, got {len(ptm.shape)}"
            )

    @abc.abstractmethod
    def reset(self, *qubits):
        """
        Reset qubits to ground state inline.

        Parameters
        ----------
        q1, ..., qN: str
            Qubits to reset
        """
        self._validate_qubits(qubits)

    @abc.abstractmethod
    def diagonal(self):
        """Compute the diagonal elements of the density matrix.

        Returns
        -------
        array
        """
        pass

    @abc.abstractmethod
    def trace(self):
        """Compute the trace of the density matrix.

        Returns
        -------
        float
        """
        pass

    @abc.abstractmethod
    def partial_trace(self, *qubits):
        """Trace out all of the qubits in the state, except mentioned in arguments.

        Parameters
        ----------
        q1, ..., qN: hashable
            Qubits to leave in the resulting state.

        Returns
        -------
        State
        """
        self._validate_qubits(qubits)

    @abc.abstractmethod
    def meas_prob(self, qubit):
        """Computes measurement probabilities of each of the possible measurement
        outcomes for a qubit.

        Measurement probabilities of 0, 1,... are returned in a formed of an array .
        If the state is not normalized (trace :math:`< 1`), resulting array is also not
        normalized. If normalized, can be supplied as `p` argument to
        :func:`numpy.random.choice` to get a random measurement result.

        Parameters
        ----------
        qubit: hashable
            Qubit tag

        Returns
        -------
        array of float
        """
        self._validate_qubits([qubit])

    @abc.abstractmethod
    def renormalize(self):
        """Rescale the state to the trace :math:`= 1`.

        Returns
        -------
        tr: int
            Trace of the density matrix before rescaling.
        """
        pass

    @abc.abstractmethod
    def copy(self):
        """Return copy of this state.

        Returns
        -------
        State
        """
        pass

    def __copy__(self):
        return self.copy()

    def exp_value(self, operator, sigma_dict=None):
        r"""
        Return an expectation value of an operator :math:`\hat{\mathcal{O}}`
        (:math:`\text{tr} \hat{\rho} \hat{\mathcal{O}}`).

        Parameters
        ----------
        operator : numpy.array or str
            An operator (TODO: elaborate)

            If string provided, computes an expectation value of the
            correspondent Pauli. Must have the same length, as the number of
            qubits in the state.
        sigma_dict: dict or None
            A dictionary of Pauli matrices or other operators, that are contained in
            `operator` and denoted by a single symbol.
            Default is `quantumsim.   algebra.sigma`.

        Returns
        -------
        float
            Expectation value for each of the measurement operators, defined in
        """
        einsum_args = []
        sigma_dict = sigma_dict or sigma
        n = len(self.qubits)
        if isinstance(operator, str):
            if n != len(operator):
                raise ValueError(
                    "Operator string must have the same length as "
                    "a number of qubits in the state"
                )
            try:
                sigmas = [sigma_dict[ch.upper()] for ch in operator]
            except KeyError as ex:
                raise ValueError("sigma_dict does not contain a key specified") from ex
            for i, s in enumerate(sigmas):
                einsum_args.append(s)
                einsum_args.append([i, n + i])
        else:
            einsum_args.append(operator)
            einsum_args.append(list(range(2 * n)))
        einsum_args.append(self.to_pv())
        einsum_args.append([2 * n + i for i in range(n)])
        for i, basis in enumerate(self.bases):
            einsum_args.append(basis.vectors)
            einsum_args.append([2 * n + i, n + i, i])
        return np.einsum(*einsum_args, optimize="greedy")

    def _validate_qubits(self, qubits):
        qubits_set = set(qubits)
        if len(qubits_set) < len(qubits):
            raise ValueError("Qubit tags can't repeat")
        absent_qubits = qubits_set - set(self.qubits)
        if len(absent_qubits) > 0:
            raise ValueError(
                f"Qubits {', '.join(absent_qubits)} are not present in " f"the state"
            )

    # noinspection PyMethodMayBeStatic
    def _validate_ptm_shape(self, ptm, target_shape, name):
        if ptm.shape != target_shape:
            raise ValueError(f"`{name}` shape must be {target_shape}, got {ptm.shape}")
