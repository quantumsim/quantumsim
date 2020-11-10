from itertools import product

import numpy as np
import xarray as xr

from quantumsim.algebra import sigma
from .. import bases
from ..pauli_vectors import PauliVectorBase
import numpy as np


class State:
    """
    Parameters
    ----------
    qubits : list of hashable
        Tags of qubits in state
    dim : int
        Hilbert dimensionality of a single-qubit subspace
    pauli_vector_class : class or None
        A class to store the system state. Must be a derivative of
        :class:`quantumsim.pauli_vectors.pauli_vector.PauliVectorBase`.
    """

    def __init__(self, qubits, *, dim=2, pauli_vector_class=None, pauli_vector=None):
        self.qubits = list(qubits)
        if pauli_vector is not None:
            self.pauli_vector = pauli_vector
        else:
            bases_ = (bases.general(dim).subbasis([0]),) * len(self.qubits)
            self.pauli_vector = self._pv_cls(pauli_vector_class)(bases_)

    @classmethod
    def from_dm(cls, qubits, dm, bases, *, pauli_vector_class=None, force=False):
        """
        Constructs a new State from an existing Pauli vector array and bases.

        Parameters
        ----------
        qubits: list of hashable
            Tags of the qubits.
        dm: array
            Density matrix of the state.
        bases: list of quantumsim.bases.PauliBasis
            Bases to store the state.
        pauli_vector_class: class, optional
            Class used for storage of Pauli vector
        force : bool
            By default creation of too large density matrix (more than
            :math:`2^22` elements currently) is not allowed. Set this to `True`
            if you know what you are doing.

        Returns
        -------
        State
        """
        pauli_vector = cls._pv_cls(pauli_vector_class).from_dm(dm, bases, force=force)
        return cls(qubits, dim=pauli_vector.dim_hilbert, pauli_vector=pauli_vector)

    @classmethod
    def from_pv(cls, qubits, pv, bases, *, pauli_vector_class=None, force=False):
        """
        Constructs a new State from an existing Pauli vector array and bases.

        Parameters
        ----------
        qubits: list of hashable
            Tags of the qubits
        pv: array
            Pauli vector
        bases: list of quantumsim.bases.PauliBasis
            Bases of the `pv`
        pauli_vector_class: class, optional
            Class used for storage of Pauli vector
        force : bool
            By default creation of too large density matrix (more than
            :math:`2^22` elements currently) is not allowed. Set this to `True`
            if you know what you are doing.

        Returns
        -------
        State
        """
        pauli_vector = cls._pv_cls(pauli_vector_class).from_dm(bases, pv, force=force)
        return cls(qubits, dim=pauli_vector.dim_hilbert, pauli_vector=pauli_vector)

    def to_dm(self):
        """
        Returns a density matrix, that corresponds to this state.

        Returns
        -------
        array
        """
        return self.pauli_vector.to_dm()

    def to_pv(self):
        """
        Returns a Pauli vector, that corresponds to this state, in the internal basis of
        the state (can be obtained via `State.bases_in` and `State.bases_out`
        properties).

        Returns
        -------
        array
        """
        return self.pauli_vector.to_pv()

    @property
    def bases(self):
        return self.pauli_vector.bases

    @staticmethod
    def _pv_cls(pauli_vector_class):
        """

        Parameters
        ----------
        pauli_vector_class: class or None

        Returns
        -------
        class
            A resolved subclass of PauliVectorBase
        """
        if pauli_vector_class is None:
            from ..pauli_vectors import Default
            return Default
        else:
            if not issubclass(pauli_vector_class, PauliVectorBase):
                raise ValueError(
                    "pauli_vector_class must be a subclass of "
                    "quantumsim.pauli_vectors.PauliVectorBase")
            return pauli_vector_class

    def __copy__(self):
        return self.copy()

    def copy(self):
        return State(self.qubits, pauli_vector=self.pauli_vector.copy())

    def exp_value(self, operator, sigma_dict=None):
        """

        Parameters
        ----------
        operator : numpy.array or str
            An operator (TODO: elaborate)

            If string provided, computes an expectation value of the
            correspondent Pauli. Must have the same length, as the number of
            qubits in the state.
        sigma_dict: dict or None
            A dictionary of Pauli matrices or other operators, that are containded in
            `operator` and denoted by a single symbol.
            Default is `quantumsim.algebra.sigma`.

        Returns
        -------
        float
            Expectation value for each of the measurement operators, defined in
        """
        einsum_args = []
        sigma_dict = sigma_dict or sigma
        n = self.pauli_vector.n_qubits
        if isinstance(operator, str):
            if n != len(operator):
                raise ValueError("Operator string must have the same length as "
                                 "a number of qubits in the state")
            try:
                sigmas = [sigma_dict[ch.upper()] for ch in operator]
            except KeyError as ex:
                raise ValueError("sigma_dict does not contain a key specified") from ex
            for i, s in enumerate(sigmas):
                einsum_args.append(s)
                einsum_args.append([i, n+i])
        else:
            einsum_args.append(operator)
            einsum_args.append(list(range(2*n)))
        einsum_args.append(self.pauli_vector.to_pv())
        einsum_args.append([2*n+i for i in range(n)])
        for i, basis in enumerate(self.pauli_vector.bases):
            einsum_args.append(basis.vectors)
            einsum_args.append([2*n+i, n+i, i])
        return np.einsum(*einsum_args, optimize=True)

    def trace(self):
        return self.pauli_vector.trace()

    def renormalize(self):
        self.pauli_vector.renormalize()

    @property
    def diagonal(self):
        diag = self.pauli_vector.diagonal()

        bases_labels = (basis.superbasis.computational_subbasis().labels
                        for basis in self.pauli_vector.bases)

        def tuple_to_string(tup):
            state = "".join(str(x) for x in tup)
            return state

        state_labels = [tuple_to_string(label)
                        for label in product(*bases_labels)]

        outcome = xr.DataArray(
            data=diag,
            dims=["state_label"],
            coords={"state_label": state_labels})
        outcome.name = "state_diags"
        return outcome

    @property
    def density_matrix(self):
        density_mat = self.pauli_vector.to_dm()

        bases_labels = (basis.superbasis.computational_subbasis().labels
                        for basis in self.pauli_vector.bases)

        def tuple_to_string(tup):
            state = "".join(str(x) for x in tup)
            return state

        state_labels = [tuple_to_string(label)
                        for label in product(*bases_labels)]

        outcome = xr.DataArray(
            data=density_mat,
            dims=["row_state_label", "col_state_label"],
            coords={"row_state_label": state_labels,
                    "col_state_label": state_labels})
        outcome.name = "state_density_mat"
        return outcome

    def partial_trace(self, *qubits):
        """Traces out all qubits, except provided, and returns the resulting
        state.

        Parameters
        ----------
        q0, q1, ... : str
            Names of qubits to preserve in the state.
        """
        return State(qubits, pauli_vector=self.pauli_vector.partial_trace(
            *[self.qubits.index(q) for q in qubits]))

    def meas_prob(self, qubit):
        """
        Returns an array of probabilities to measure each state of a `qubit`.
        May not be normalized to 1.

        Parameters
        ----------
        qubit: hashable
            Tag of a qubit

        Returns
        -------
        array
        """
        try:
            return self.pauli_vector.meas_prob(self.qubits.index(qubit))
        except ValueError as ex:
            raise ValueError(f'Qubit {qubit} is not in the state') from ex
