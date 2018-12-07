# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import collections


class PauliBasis():
    """Defines a Pauli basis [1]_ . TODO.

    References
    ----------
    .. [1] A "Pauli basis" is an orthonormal basis (w.r.t
        :math:`\\langle A, B \\rangle = \\text{Tr}(A \\cdot B^\\dagger)`)
        for a space of Hermitian matrices.
    "a tensor B of shape (dim_pauli, dim_hilbert, dim_hilbert)"
    "read as a vector of matrices"
    "must satisfy Tr(B[i] @ B[j]) = delta(i, j)"
    """

    def __init__(self, vectors, labels, superbasis=None, subsys_dims=None):
        if vectors.shape[1] != vectors.shape[2]:
            raise ValueError(
                "Pauli basis vectors must be square matrices, got shape {}x{}"
                .format(vectors.shape[1], vectors.shape[2]))

        self.vectors = vectors
        self.labels = labels
        self._superbasis = superbasis
        self.subsys_dims = subsys_dims or np.array([self.vectors.shape[:2]])

        # TODO: rename? Or may be refactor to avoid needs to hint?
        self.computational_basis_vectors = np.einsum(
            "xii -> ix", self.vectors, optimize=True)

        # make hint on how to efficiently
        # extract the diagonal
        self.comp_basis_indices = {
            i: self._to_unit_vector(cb)
            for i, cb in enumerate(self.computational_basis_vectors)}

        # make hint on how to trace
        traces = np.einsum("xii", self.vectors, optimize=True) / \
            np.sqrt(self.dim_hilbert)

        self.trace_index = self._to_unit_vector(traces)

    def __mul__(self, pauli_basis):
        expanded_vectors = np.kron(self.vectors, pauli_basis.vectors)

        expanded_labels = np.array(
            [label_1+label_2 for label_1 in self.labels for label_2 in pauli_basis.labels])

        expanded_superbasis = None

        expanded_subsys_dims = np.concatenate(
            (self.subsys_dims, pauli_basis.subsys_dims))

        return PauliBasis(expanded_vectors, expanded_labels, expanded_superbasis, expanded_subsys_dims)

    @property
    def dim_hilbert(self):
        return self.vectors.shape[1]

    @property
    def dim_pauli(self):
        return self.vectors.shape[0]

    @property
    def subsys_dims(self):
        return self.subsys_dims.shape[0]

    @property
    def subsys_pauli_dims(self):
        return self.subsys_dims[:, 0]

    @property
    def subsys_hilbert_dims(self):
        return self.subsys_dims[:, 1]

    @property
    def num_subsys(self):
        return self.subsys_dims.shape[0]

    @property
    def superbasis(self):
        return self._superbasis or self

    def subbasis(self, indices):
        """
        return a subbasis of this basis
        """
        return PauliBasis(self.vectors[indices],
                          [self.labels[i] for i in indices], self)

    def classical_subbasis(self):
        idxes = [idx
                 for st, idx in self.comp_basis_indices.items()
                 if idx is not None]
        return self.subbasis(idxes)

    def hilbert_to_pauli_vector(self, rho):
        return np.einsum("xab, ba -> x", self.vectors, rho, optimize=True)

    def is_orthonormal(self):
        i = np.einsum("xab, yba -> xy", self.vectors,
                      self.vectors, optimize=True)
        assert np.allclose(i, np.eye(self.dim_pauli))

    @staticmethod
    def _to_unit_vector(v):
        if np.allclose(np.sum(v), 1):
            rounded = np.round(v, 8)
            nz, = np.nonzero(rounded)
            if len(nz) == 1:
                return nz[0]
        return None

    def __repr__(self):
        s = "<{} d_hilbert={}, d_pauli={}, {}>"

        if self.labels:
            bvn_string = " ".join(self.labels)
        else:
            bvn_string = "unnamed basis"

        return s.format(
            self.__class__.__name__,
            self.dim_hilbert,
            self.dim_pauli,
            bvn_string)
