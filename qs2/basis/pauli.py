# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt
"""General classes for Pauli bases and PTMs"""

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

    def __init__(self, vectors, labels, superbasis=None):
        if vectors.shape[1] != vectors.shape[2]:
            raise ValueError(
                "Pauli basis vectors must be square matrices, got shape {}x{}"
                .format(vectors.shape[1], vectors.shape[2]))

        self.vectors = vectors
        self.labels = labels
        self._superbasis = superbasis

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

    @property
    def dim_hilbert(self):
        return self.vectors.shape[1]

    @property
    def dim_pauli(self):
        return self.vectors.shape[0]

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


class PTM:
    def __init__(self):
        """
        A Pauli transfer matrix. ABC
        """

        "The Hilbert space dimension on which the PTM operates"
        raise NotImplementedError

    def get_matrix(self, in_basis, out_basis=None):
        """Return the matrix representation of this PTM in the basis given.

        If `out_basis` is None, `out_basis = in_basis` is assumed.

        If `out_basis` spans only a subspace, projection on that subspace is
        implicit.
        """
        raise NotImplementedError

    def __add__(self, other):
        if not isinstance(other, PTM):
            raise ValueError("Cannot add PTM to non-ptm")
        if isinstance(other, LinearCombPTM):
            return LinearCombPTM(self.elements + other.elements)
        else:
            return LinearCombPTM({self: 1, other: 1})

    def __mul__(self, scalar):
        return LinearCombPTM({self: scalar})

    # vector space boiler plate...

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        if isinstance(other, LinearCombPTM):
            return LinearCombPTM(self.elements - other.elements)
        else:
            return LinearCombPTM(self.elements - collections.Counter([other]))

    def __rsub__(self, other):
        return other.__sub__(self)

    def __matmul__(self, other):
        return ProductPTM([self, other])

    def _check_basis_op_consistency(self, basis_out, basis_in):
        if self.op.shape != (basis_out.dim_hilbert, basis_in.dim_hilbert):
            raise ValueError(
                "Basis Hilbert dimention is incompatible with `self.op`.\n"
                "basis_out Hilbert dimension must be {}, got {}\n"
                "basis_in Hilbert dimension must be {}, got {}"
                    .format(self.op.shape[0], basis_out.dim_hilbert,
                            self.op.shape[1], basis_in.dim_hilbert))

    @staticmethod
    def _check_basis_ptm_consistency(ptm, basis):
        if ptm.shape[0] != ptm.shape[1]:
            raise ValueError(
                "ptm must be a square matrix, got shape {}".format(ptm.shape))

        if ptm.shape[0] != basis.dim_pauli:
            raise ValueError(
                "basis Pauli dimention is incompatible with `ptm`.\n"
                "basis Pauli dimension must be {}, got {}\n"
                    .format(ptm.shape[0], basis.dim_pauli))


class ExplicitBasisPTM(PTM):
    def __init__(self, ptm, basis):
        self._check_basis_ptm_consistency(ptm, basis)
        self.ptm = ptm
        self.basis = basis

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        result = np.einsum("xab, yba, yz, zcd, wdc -> xw",
                           basis_out.basisvectors,
                           self.basis.basisvectors,
                           self.ptm,
                           self.basis.basisvectors,
                           basis_in.basisvectors, optimize=True).real

        return result


class LinearCombPTM(PTM):
    def __init__(self, elements):
        """A linear combination of other PTMs. Should usually not be
        instantiated by hand, but created as a result of adding or scaling
        PTMs.
        """

        # check dimensions
        dimensions_set = set(p.dim_hilbert for p in elements.keys())
        if len(dimensions_set) > 1:
            raise ValueError(
                "Cannot create linear combination: incompatible dimensions: {}"
                    .format(dimensions_set))

        self.dim_hilbert = dimensions_set.pop()

        # float defaultdict of shape {ptm: coefficient, ...}
        self.elements = collections.Counter(elements)

    def get_matrix(self, in_basis, out_basis=None):
        return sum(c * p.get_matrix(in_basis, out_basis)
                   for p, c in self.elements.items())

    def __mul__(self, scalar):
        return LinearCombPTM({p: scalar * c for p, c in self.elements.items()})

    def __add__(self, other):
        if not isinstance(other, PTM):
            raise ValueError("cannot add PTM to non-ptm")
        if isinstance(other, LinearCombPTM):
            return LinearCombPTM(self.elements + other.elements)
        else:
            return LinearCombPTM(self.elements + collections.Counter([other]))


class ProductPTM(PTM):
    def __init__(self, elements):
        """A product of other PTMs. Should usually not be instantiated by hand,
        but created as a result of multiplying PTMs.

        Parameters
        ----------
        elements : list of :class:`PTM` derivatives
            List of factors. Will be multiplied in order
            `elements[n-1] @ ... @ elements[0]`.
        """

        # check dimensions
        dimensions_set = set(p.dim_hilbert for p in elements)
        if len(dimensions_set) > 1:
            raise ValueError(
                "cannot create product: incompatible dimensions: {}"
                    .format(dimensions_set))

        elif len(dimensions_set) == 1:
            self.dim_hilbert = dimensions_set.pop()
        else:
            self.dim_hilbert = None

        self.elements = elements

    def get_matrix(self, basis_in, basis_out=None):
        # FIXME: product is always formed in the complete basis, which might be
        # not efficient
        if basis_out is None:
            basis_out = basis_in

        if basis_in.dim_hilbert != basis_out.dim_hilbert:
            raise ValueError(
                "`basis_in` and `basis_out` must have equal Hilbert dimensions"
                ", got {} and {} correspondingly."
                    .format(basis_in.dim_hilbert, basis_out.dim_hilbert))

        if self.dim_hilbert and basis_in.dim_hilbert != self.dim_hilbert:
            raise ValueError(
                "`basis_in` and `self` must have equal Hilbert dimensions"
                ", got {} and {} correspondingly."
                    .format(basis_in.dim_hilbert, basis_out.dim_hilbert))

        complete_basis = GeneralBasis(basis_in.dim_hilbert)
        result = np.eye(complete_basis.dim_pauli)
        for pi in self.elements:
            pi_mat = pi.get_matrix(complete_basis)
            result = pi_mat @ result

        trans_mat_in = np.einsum(
            "xab, yba",
            complete_basis.basisvectors,
            basis_in.basisvectors, optimize=True)
        trans_mat_out = np.einsum(
            "xab, yba",
            basis_out.basisvectors,
            complete_basis.basisvectors, optimize=True)

        return (trans_mat_out @ result @ trans_mat_in).real

    def __matmul__(self, other):
        if isinstance(other, ProductPTM):
            return ProductPTM(self.elements + other.elements)
        else:
            return ProductPTM(self.elements + [other])

    def __rmatmul__(self, other):
        return other.__matmul__(self)


class ConjunctionPTM(PTM):
    def __init__(self, op):
        """The PTM describing conjunction with unitary operator `op`, i.e.

        .. math::
            \\rho^\\prime = P(\\rho) =
            \\text{op} \\cdot \\rho \\cdot \\text{op}^\\dagger

        Typical usage: `op` describes the unitary time evolution of a system
        described by :math:`\\rho`.

        Parameters
        ----------
        op : 2D array
            A matrix given in computational basis.
        """
        self.op = np.array(op)
        self.dim_hilbert = self.op.shape[0]

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        self._check_basis_op_consistency(basis_out, basis_in)

        st_out = basis_out.basisvectors
        st_in = basis_in.basisvectors

        result = np.einsum("xab, bc, ycd, ad -> xy", st_out, self.op, st_in,
                           self.op.conj(), optimize=True)

        assert np.allclose(result.imag, 0)

        return result.real

    def embed_hilbert(self, new_dim_hilbert, mp=None):
        if mp is None:
            mp = range(min(self.dim_hilbert, new_dim_hilbert))

        proj = np.zeros((self.dim_hilbert, new_dim_hilbert))
        for i, j in enumerate(mp):
            proj[i, j] = 1

        new_op = np.eye(new_dim_hilbert) - proj.T @ proj + \
                 proj.T @ self.op @ proj

        return ConjunctionPTM(new_op)


class PLMIntegrator:
    def __init__(self, plm):
        self.plm = plm
        self.full_basis = GeneralBasis(self.plm.dim_hilbert)
        self.lindbladian_mat = self.plm.get_matrix(self.full_basis)

        self.e, self.v = np.linalg.eig(self.lindbladian_mat)
        self.vinv = np.linalg.inv(self.v)

    def get_ptm(self, power):
        p = self.v @ np.diag(np.exp(power * self.e)) @ self.vinv

        return ExplicitBasisPTM(p, self.full_basis)


class AdjunctionPLM(PTM):
    def __init__(self, op):
        """The PLM (Pauli Liouvillian Matrix) describing adjunction with
        operator `op`, i.e.

        Typical usage: op is a Hamiltonian, the PTM describes the infinitesimal
        evolution of rho.

        .. math::
            \\rho^\\prime = P(\\rho) =
            1i \\left(\\text{op}\\cdot\\rho - \\rho\\cdot\\text{op} \\right)

        Parameters
        ----------
        op : 2D array
            A matrix given in computational basis.
        """

        self.op = np.array(op)
        self.dim_hilbert = op.shape[0]

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        self._check_basis_op_consistency(basis_out, basis_in)

        st_out = basis_out.basisvectors
        st_in = basis_in.basisvectors

        result = np.einsum("xab, bc, yca -> xy", st_out, self.op, st_in,
                           optimize=True)

        # taking the real part implements the two parts of the commutator
        return result.imag


class LindbladPLM(PTM):
    def __init__(self, op):
        """The PLM describing the Lindblad superoperator of `op`, i.e.

        .. math::
            \\rho^\\prime = P(\\rho) =
            \\text{op}\\cdot\\rho\\cdot\\text{op}^\\dagger -
            \\frac{1}{2} \\left\\{
                \\text{op}^\\dagger\\cdot\\text{op}, \\rho
            \\right\\}

        Typical usage: op is a decay operator.
        """
        self.op = op
        self.dim_hilbert = op.shape[0]

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        self._check_basis_op_consistency(basis_out, basis_in)

        st_out = basis_out.basisvectors
        st_in = basis_in.basisvectors

        result = np.einsum("xab, bc, ycd, ad -> xy", st_out, self.op, st_in,
                           self.op.conj(), optimize=True)
        result -= 0.5 * np.einsum("xab, cb, cd, yda -> xy", st_out,
                                  self.op.conj(), self.op, st_in,
                                  optimize=True)

        result -= 0.5 * np.einsum("xab, ybc, dc, da -> xy", st_out, st_in,
                                  self.op.conj(), self.op, optimize=True)

        return result.real


class RotateXPTM(ConjunctionPTM):
    def __init__(self, angle):
        s, c = np.sin(angle / 2), np.cos(angle / 2)
        super().__init__([[c, -1j * s], [-1j * s, c]])


class RotateYPTM(ConjunctionPTM):
    def __init__(self, angle):
        s, c = np.sin(angle / 2), np.cos(angle / 2)
        super().__init__([[c, -s], [s, c]])


class RotateZPTM(ConjunctionPTM):
    def __init__(self, angle):
        z = np.exp(-.5j * angle)
        super().__init__([[z, 0], [0, z.conj()]])


class AmplitudePhaseDampingPTM(ProductPTM):
    def __init__(self, gamma, lamda):
        e0 = [[1, 0], [0, np.sqrt(1 - gamma)]]
        e1 = [[0, np.sqrt(gamma)], [0, 0]]
        amp_damp = ConjunctionPTM(e0) + ConjunctionPTM(e1)

        e0 = [[1, 0], [0, np.sqrt(1 - lamda)]]
        e1 = [[0, 0], [0, np.sqrt(lamda)]]
        ph_damp = ConjunctionPTM(e0) + ConjunctionPTM(e1)

        super().__init__([amp_damp, ph_damp])


class TwoPTM:
    def __init__(self, dim0, dim1):
        pass

    def get_matrix(self, bases_in, bases_out):
        pass

    def multiply(self, subspace, process):
        pass

    def multiply_two(self, other):
        pass


class TwoPTMProduct(TwoPTM):
    def __init__(self, elements=None):
        """A product of single- and two-qubit process matrices.

        Parameters
        ----------

        elements : list of tuples
            A list of `(bit0, bit1, two_ptm)` or `(bit, single_ptm)`,
            where `bit0`, `bit1` in [0, 1]
        """
        self.elements = elements
        if elements is None:
            self.elements = []

    def get_matrix(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in

        # internally done in full basis
        complete_basis0 = GeneralBasis(bases_in[0].dim_hilbert)
        complete_basis1 = GeneralBasis(bases_in[1].dim_hilbert)

        complete_basis = [complete_basis0, complete_basis1]

        result = np.eye(complete_basis0.dim_pauli * complete_basis1.dim_pauli)

        result = result.reshape((
            complete_basis0.dim_pauli,
            complete_basis1.dim_pauli,
            complete_basis0.dim_pauli,
            complete_basis1.dim_pauli,
        ))

        for bits, pt in self.elements:
            if len(bits) == 1:
                # single PTM
                bit = bits[0]
                pmat = pt.get_matrix(complete_basis[bit])
                if bit == 0:
                    result = np.einsum(pmat, [0, 10], result, [10, 1, 2, 3],
                                       [0, 1, 2, 3], optimize=True)
                if bit == 1:
                    result = np.einsum(pmat, [1, 10], result, [0, 10, 2, 3],
                                       [0, 1, 2, 3], optimize=True)

            elif len(bits) == 2:
                # double ptm
                pmat = pt.get_matrix(
                    [complete_basis[bits[0]], complete_basis[bits[1]]])
                if tuple(bits) == (0, 1):
                    result = np.einsum(
                        pmat, [0, 1, 10, 11],
                        result, [10, 11, 2, 3], optimize=True)
                elif tuple(bits) == (1, 0):
                    result = np.einsum(
                        pmat, [1, 0, 11, 10],
                        result, [10, 11, 2, 3], optimize=True)
                else:
                    raise ValueError()
            else:
                raise ValueError()

        # return the result in the right basis, hell yeah
        result = np.einsum(
            bases_out[0].basisvectors, [0, 21, 22],
            complete_basis[0].basisvectors, [11, 22, 21],
            bases_out[1].basisvectors, [1, 23, 24],
            complete_basis[1].basisvectors, [12, 24, 23],
            result, [11, 12, 13, 14],
            complete_basis[0].basisvectors, [13, 25, 26],
            bases_in[0].basisvectors, [2, 26, 25],
            complete_basis[1].basisvectors, [14, 27, 28],
            bases_in[1].basisvectors, [3, 28, 27], [0, 1, 2, 3], optimize=True)

        return result.real


class TwoKrausPTM(TwoPTM):
    def __init__(self, unitary):
        """Create a two-subsystem process matrix from a unitary.
        The unitary has to have shape of form [x, y, x, y],
        where x(y) is the dimension of the first (second) subsystem.

        Parameters
        ----------
        unitary : 4D array
            The unitary. Has to have a shape of form `(x, y, x, y)`,
            where x(y) is the dimension of the first (second) subsystem.

        """
        if len(unitary.shape) != 4 or unitary.shape[0:2] != unitary.shape[2:4]:
            raise ValueError("unitary has wrong shape: {}"
                             .format(unitary.shape))

        self.unitary = unitary
        self.dim_hilbert = unitary.shape[0:2]

    def get_matrix(self, bases_in, bases_out=None):
        """Return the process matrix in the basis

        Parameters
        ----------
        bases_in : tuple of two :class:`~PauliBasis` derivatives
            Input bases in the form `(basis0_in, basis1_in)`.
        bases_out: tuple of two :class:`~PauliBasis` derivatives or None
            Output bases in the form `(basis0_in, basis1_in)`.
            If `None`, assumed equal to `bases_in`.
        """
        st0i = bases_in[0].basisvectors
        st1i = bases_in[1].basisvectors

        if bases_out is None:
            st0o, st1o = st0i, st1i
        else:
            st0o = bases_out[0].basisvectors
            st1o = bases_out[1].basisvectors

        kraus = self.unitary

        # very nice contraction :D
        return np.einsum(st0o, [20, 1, 3], st1o, [21, 2, 4],
                         kraus, [3, 4, 5, 6],
                         st0i, [22, 5, 7], st1i, [23, 6, 8],
                         kraus.conj(), [1, 2, 7, 8],
                         [20, 21, 22, 23], optimize=True).real


class CPhaseRotationPTM(TwoKrausPTM):
    def __init__(self, angle=np.pi):
        u = np.diag([1, 1, 1, np.exp(1j * angle)]).reshape(2, 2, 2, 2)
        super().__init__(u)


class TwoPTMExplicit(TwoPTM):
    def __init__(self, ptm, basis0, basis1):
        pass
