from itertools import product

import numpy as np
import xarray as xr

from quantumsim.algebra import sigma

from .. import bases
from ..pauli_vectors import PauliVectorBase


class State:
    """
    Parameters
    ----------
    qubits : list of str
        Names of qubits in state
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
            if pauli_vector_class is None:
                from ..pauli_vectors import Default

                self.pauli_vector = Default(bases_)
            else:
                if not issubclass(pauli_vector_class, PauliVectorBase):
                    raise ValueError(
                        "pauli_vector_class must be a subclass of "
                        "quantumsim.pauli_vectors.PauliVectorBase"
                    )
                self.pauli_vector = pauli_vector_class(bases_)

    def __copy__(self):
        return self.copy()

    def copy(self):
        return State(self.qubits, pauli_vector=self.pauli_vector.copy())

    def exp_value(self, operator):
        """

        Parameters
        ----------
        operator : numpy.array or str
            An operator (TODO: elaborate)

            If string provided, computes an expectation value of the
            correspondent Pauli. Must have the same length, as the number of
            qubits in the state.

        Returns
        -------
        float
            Expectation value for each of the measurement operators, defined in
        """
        einsum_args = []
        n = self.pauli_vector.n_qubits
        if isinstance(operator, str):
            if n != len(operator):
                raise ValueError(
                    "operator string must have the same length "
                    "as number of qubits in the state"
                )
            try:
                sigmas = [sigma[ch.upper()] for ch in operator]
            except KeyError as ex:
                raise ValueError(
                    "operator string must contain only I, X, " "Y or Z"
                ) from ex
            for i, s in enumerate(sigmas):
                einsum_args.append(s)
                einsum_args.append([i, n + i])
        else:
            einsum_args.append(operator)
            einsum_args.append(list(range(2 * n)))
        einsum_args.append(self.pauli_vector.to_pv())
        einsum_args.append([2 * n + i for i in range(n)])
        for i, basis in enumerate(self.pauli_vector.bases):
            einsum_args.append(basis.vectors)
            einsum_args.append([2 * n + i, n + i, i])
        print(einsum_args)
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
        return State(qubits, pauli_vector=self.pauli_vector.partial_trace(*[self.qubits.index(q) for q in qubits]),)
