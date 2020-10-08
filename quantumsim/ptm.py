# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import collections
from itertools import chain
from functools import reduce

# The transformation matrix between the two bases. Its essentially a Hadamard,
# so its its own inverse.
basis_transformation_matrix = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)]])

single_tensor = np.array([[[1, 0], [0, 0]],
                          np.sqrt(0.5) * np.array([[0, 1], [1, 0]]),
                          np.sqrt(0.5) * np.array([[0, -1j], [1j, 0]]),
                          [[0, 0], [0, 1]]])

double_tensor = np.kron(single_tensor, single_tensor)

_ptm_basis_vectors_cache = {}


def general_ptm_basis_vector(n):
    """The vector of 'Pauli matrices' in dimension n.
    First the n diagonal matrices, then
    the off-diagonals in x-like, y-like pairs
    """

    if n in _ptm_basis_vectors_cache:
        return _ptm_basis_vectors_cache[n]
    else:

        basis_vector = []

        for i in range(n):
            v = np.zeros((n, n), np.complex)
            v[i, i] = 1
            basis_vector.append(v)

        for i in range(n):
            for j in range(i):
                # x-like
                v = np.zeros((n, n), np.complex)
                v[i, j] = np.sqrt(0.5)
                v[j, i] = np.sqrt(0.5)
                basis_vector.append(v)

                # y-like
                v = np.zeros((n, n), np.complex)
                v[i, j] = 1j * np.sqrt(0.5)
                v[j, i] = -1j * np.sqrt(0.5)
                basis_vector.append(v)

        basis_vector = np.array(basis_vector)

        _ptm_basis_vectors_cache[n] = basis_vector

    return basis_vector


def to_0xy1_basis(ptm, general_basis=False):
    """Transform a Pauli transfer in the "usual" basis (0xyz) [1]_ ,
    to the 0xy1 basis which is required by
    :meth:`~SparseDM.apply_ptm`.

    If general_basis is True, transform to the 01xy basis, which is the
    two-qubit version of the general basis defined by
    :func:`~general_ptm_basis_vector`.

    Parameters
    ----------
    ptm : 2D array
        The input transfer matrix in 0xyz basis. Can be 4x4, 4x3 or
        3x3 matrix of real numbers.

        If 4x4, the first row must be (1,0,0,0). If 4x3, this row is considered
        to be omitted. If 3x3, the transformation is assumed to be unitary,
        thus it is assumed that the first column is also (1,0,0,0) and was
        omitted.
    general_basis : boolean
        Whether to transform to 01xy basis

    Returns
    -------
    result : 2D array
        Resulting Pauli transfer matrix in 0xy1 or 01xy basis.

    References
    ----------
    .. [1] Daniel Greenbaum, Introduction to Quantum Gate Set Tomography,
        http://arxiv.org/abs/1509.02921v1
    """

    ptm = np.array(ptm)

    if ptm.shape == (3, 3):
        ptm = np.hstack(([[0], [0], [0]], ptm))
    if ptm.shape == (3, 4):
        ptm = np.vstack(([1, 0, 0, 0], ptm))
    if not ptm.shape == (4, 4):
        raise ValueError("Input PTM must be 3x3, 3x4 or 4x4 array")

    if not np.allclose(ptm[0, :], [1, 0, 0, 0]):
        raise ValueError("First row of PTM must be [1, 0, 0, 0]")

    if general_basis:
        result = ExplicitBasisPTM(
            ptm, PauliBasis_ixyz()).get_matrix(
            GeneralBasis(2))
    else:
        result = ExplicitBasisPTM(
            ptm, PauliBasis_ixyz()).get_matrix(
            PauliBasis_0xy1())

    return result


def to_0xyz_basis(ptm):
    """Transform a Pauli transfer in the 0xy1 basis [1]_ ,
    to the the usual 0xyz. The inverse of :func:`~to_0xy1_basis`.

    Parameters
    ----------
    ptm: 2D array
        The input transfer matrix in 0xy1 basis. Must be 4x4.

    References
    ----------
    .. [1] Daniel Greenbaum, Introduction to Quantum Gate Set Tomography,
        http://arxiv.org/abs/1509.02921v1
    """

    ptm = np.array(ptm)
    if ptm.shape == (4, 4):
        trans_mat = basis_transformation_matrix
        return np.dot(trans_mat, np.dot(ptm, trans_mat))
    elif ptm.shape == (16, 16):
        trans_mat = np.kron(
            basis_transformation_matrix,
            basis_transformation_matrix)
        return np.dot(trans_mat, np.dot(ptm, trans_mat))
    else:
        raise ValueError(
            "Wrong dimension of input PTM, must be one- or two-Pauli "
            "transfer matrix (4x4 or 16x16 array).")


def hadamard_ptm(general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis, representing perfect
    unitary Hadamard (Rotation around the :math:`(x+z)/\\sqrt{2}` axis by
    :math:`\\pi`).
    """
    u = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
    if general_basis:
        pb = GeneralBasis(2)
    else:
        pb = PauliBasis_0xy1()
    return ConjunctionPTM(u).get_matrix(pb)


def amp_ph_damping_ptm(gamma, lamda, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis, representing amplitude
    and phase damping with parameters `gamma` and `lambda`. [1]_

    References
    ----------
    .. [1] See Nielsen & Chuang for definition.
    """

    if general_basis:
        pb = GeneralBasis(2)
    else:
        pb = PauliBasis_0xy1()

    return AmplitudePhaseDampingPTM(gamma, lamda).get_matrix(pb)


def gen_amp_damping_ptm(gamma_down, gamma_up):
    """Return a 4x4 Pauli transfer matrix, representing amplitude damping,
    including an excitation rate `gamma_up`. [1]_

    References
    ----------
    .. [1] TODO
    """
    gamma = gamma_up + gamma_down
    ptm = np.array([
        [1, 0, 0, 0],
        [0, np.sqrt((1 - gamma)), 0, 0],
        [0, 0, np.sqrt((1 - gamma)), 0],
        [2*gamma_down - gamma, 0, 0, 1 - gamma]]
    )
    return to_0xy1_basis(ptm)


def dephasing_ptm(px, py, pz):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis, representing dephasing
    (shrinking of the Bloch sphere along the principal axes), with different
    rates across the different axes. :math:`p_i/2` is the flip probability,
    so :math:`p_i = 0` corresponds to no shrinking, while :math:`p_i = 1` is
    total dephasing.
    """

    ptm = np.diag([1 - px, 1 - py, 1 - pz])
    return to_0xy1_basis(ptm)


def bitflip_ptm(p):
    ptm = np.diag([1 - p, 1, 1])
    return to_0xy1_basis(ptm)


def rotate_x_ptm(angle, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the x-axis by angle.
    """
    if general_basis:
        pb = GeneralBasis(2)
    else:
        pb = PauliBasis_0xy1()
    return RotateXPTM(angle).get_matrix(pb)


def rotate_y_ptm(angle, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the y-axis by angle.
    """
    ptm = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])

    return to_0xy1_basis(ptm, general_basis)


def rotate_z_ptm(angle, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary rotation around the z-axis by angle.
    """
    ptm = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    return to_0xy1_basis(ptm, general_basis)


def single_kraus_to_ptm_general(kraus):
    d = kraus.shape[0]
    if kraus.shape != (d, d):
        raise ValueError("kraus must be a square matrix, got shape {}"
                         .format(kraus.shape))

    st = general_ptm_basis_vector(d)

    return np.einsum("xab, bc, ycd, ad -> xy", st,
                     kraus, st, kraus.conj(), optimize=True).real


def single_kraus_to_ptm(kraus, general_basis=False):
    """Given a Kraus operator in z-basis, obtain the corresponding
    single-qubit ptm in 0xy1 basis
    """
    if general_basis:
        st = general_ptm_basis_vector(2)
    else:
        st = single_tensor
    return np.einsum("xab, bc, ycd, ad -> xy", st,
                     kraus, st, kraus.conj(), optimize=True).real


def double_kraus_to_ptm(kraus, general_basis=False):
    if general_basis:
        st = general_ptm_basis_vector(2)
    else:
        st = single_tensor

    dt = np.kron(st, st)

    return np.einsum("xab, bc, ycd, ad -> xy", dt,
                     kraus, dt, kraus.conj(), optimize=True).real


def _to_unit_vector(v):
    if np.allclose(np.sum(v), 1):
        rounded = np.round(v, 8)
        nz, = np.nonzero(rounded)
        if len(nz) == 1:
            return nz[0]
    return None


class PauliBasis:
    def __init__(self, basisvectors=None, basisvector_names=None):
        """Defines a Pauli basis [1]_ . TODO.

        References
        ----------
        .. [1] A "Pauli basis" is an orthonormal basis (w.r.t
            :math:`\\langle A, B \\rangle = \\text{Tr}(A \\cdot B^\\dagger)`)
            for a space of Hermitian matrices.
        """

        "a tensor B of shape (dim_pauli, dim_hilbert, dim_hilbert)"
        "read as a vector of matrices"
        "must satisfy Tr(B[i] @ B[j]) = delta(i, j)"
        if basisvectors is not None:
            self.basisvectors = basisvectors

        if basisvector_names is not None:
            self.basisvector_names = basisvector_names

        shape = self.basisvectors.shape

        if shape[1] != shape[2]:
            raise ValueError(
                "basisvectors.shape[1] must be equal tobasisvectors.shape[2], "
                "got {} and {} correspondingly"
                .format(shape[1], shape[2]))

        self.dim_hilbert = shape[1]
        self.dim_pauli = shape[0]

        self.superbasis = None

        self.computational_basis_vectors = np.einsum(
            "xii -> ix", self.basisvectors, optimize=True)

        # make hint on how to efficiently
        # extract the diagonal
        cbi = {i: _to_unit_vector(cb)
               for i, cb in enumerate(self.computational_basis_vectors)}

        self.comp_basis_indices = cbi

        # make hint on how to trace
        traces = np.einsum("xii", self.basisvectors, optimize=True) / \
            np.sqrt(self.dim_hilbert)

        self.trace_index = _to_unit_vector(traces)

    def get_superbasis(self):
        if self.superbasis:
            return self.superbasis
        else:
            return self

    def get_subbasis(self, idxes):
        """
        return a subbasis of this basis
        """

        bvn = [self.basisvector_names[i] for i in idxes]

        subbasis = PauliBasis(self.basisvectors[idxes], bvn)

        subbasis.superbasis = self
        return subbasis

    def get_classical_subbasis(self):
        idxes = [idx
                 for st, idx in self.comp_basis_indices.items()
                 if idx is not None]
        return self.get_subbasis(idxes)

    def hilbert_to_pauli_vector(self, rho):
        return np.einsum("xab, ba -> x", self.basisvectors, rho, optimize=True)

    def check_orthonormality(self):
        i = np.einsum("xab, yba -> xy", self.basisvectors,
                      self.basisvectors, optimize=True)
        assert np.allclose(i, np.eye(self.dim_pauli))

    def __repr__(self):
        s = "<{} d_hilbert={}, d_pauli={}, {}>"

        if self.basisvector_names:
            bvn_string = " ".join(self.basisvector_names)
        else:
            bvn_string = "unnamed basis"

        return s.format(
            self.__class__.__name__,
            self.dim_hilbert,
            self.dim_pauli,
            bvn_string)


class GeneralBasis(PauliBasis):
    def __init__(self, dim):
        """A "general" Pauli basis in the sense that is defined for every
        Hilbert space dimension.

        `GeneralBasis(2)` is the same as `PauliBasis_0xy1()`,
        but with the elements ordered differently.

        The basis matrices are:
          - Matrices with one "1" on the diagonal, followed by
          - pairs of "X" like (real) and "Y" like (imaginary) with two non-zero
            elements each.
        """
        self.basisvectors = general_ptm_basis_vector(dim)

        self.basisvector_names = []

        for i in range(dim):
            self.basisvector_names.append(str(i))

        for j in range(dim):
            for i in range(i):
                self.basisvector_names.append("X{}{}".format(i, j))
                self.basisvector_names.append("Y{}{}".format(i, j))

        super().__init__()


class PauliBasis_0xy1(PauliBasis):
    """A Pauli basis for a two-dimensional Hilbert space.
    The basis consisting of projections to 0, Pauli matrices :math:`\\sigma_x`
    and :math:`\\sigma_y`, and projection to 1, in that order.

    The Pauli basis used by older versions of quantumsim.
    """
    basisvectors = np.array([[[1, 0], [0, 0]],
                             np.sqrt(0.5) * np.array([[0, 1], [1, 0]]),
                             np.sqrt(0.5) * np.array([[0, -1j], [1j, 0]]),
                             [[0, 0], [0, 1]]])
    basisvector_names = ["0", "X", "Y", "1"]


class PauliBasis_ixyz(PauliBasis):
    """A Pauli basis for two-dimensional Hilbert spaces, the standard Pauli
    basis consisting of identity and the three Pauli matrices.
    """
    basisvectors = np.sqrt(0.5) * np.array([[[1, 0], [0, 1]],
                                            [[0, 1], [1, 0]],
                                            [[0, -1j], [1j, 0]],
                                            [[1, 0], [0, -1]]])

    basisvector_names = ["I", "X", "Y", "Z"]


class GellMannBasis(PauliBasis):
    """A Pauli basis consisting of the generalization of Pauli matrices for
    higher dimensions, the generalized Gell-Mann matrices [1]_ [2]_ .

    These matrices are Hermitian and traceless, except the first, which is the
    identity.

    `GellMannBasis(2)` is the same as `PauliBasis_ixyz()`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
    .. [2] https://en.wikipedia.org/wiki/Gell-Mann_matrices
    """

    def __init__(self, dim_hilbert):

        diag_gell_manns = [np.ones(dim_hilbert) / np.sqrt(dim_hilbert)]
        for i in range(1, dim_hilbert):
            di = np.zeros(dim_hilbert)
            di[:i] = 1
            di[i] = -i
            diag_gell_manns.append(di / np.sqrt(i * (i + 1)))

        gellmanns = []

        basisvector_names = []

        for i in range(dim_hilbert):
            for j in range(dim_hilbert):
                basisvector_names.append("γ{}{}".format(i, j))
                if i == j:
                    g = np.diag(diag_gell_manns[i])
                else:
                    g = np.zeros((dim_hilbert, dim_hilbert), np.complex)
                    if i < j:
                        g[i, j] = np.sqrt(.5)
                        g[j, i] = np.sqrt(.5)
                    else:
                        g[i, j] = 1j * np.sqrt(.5)
                        g[j, i] = -1j * np.sqrt(.5)
                gellmanns.append(g)

        gellmanns = np.array(gellmanns)

        super().__init__(gellmanns, basisvector_names)


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
    def __init__(self, ptm, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        self._check_basis_ptm_consistency(ptm, basis_in)
        self._check_basis_ptm_consistency(ptm, basis_out)
        self.ptm = ptm
        self._basis_in = basis_in
        self._basis_out = basis_out
        assert basis_in.dim_hilbert == basis_out.dim_hilbert
        self.dim_hilbert = basis_in.dim_hilbert

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        result = np.einsum("xab, yba, yz, zcd, wdc -> xw",
                           basis_out.basisvectors,
                           self._basis_out.basisvectors,
                           self.ptm,
                           self._basis_in.basisvectors,
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
            result = pi_mat@result

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

        new_op = np.eye(new_dim_hilbert) - proj.T@proj + \
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
        p = self.v @ np.diag(np.exp(power*self.e)) @ self.vinv

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

        # result = np.einsum("xab, bc, yca -> xy", st_out, self.op, st_in,
        #                   optimize=True)

        out = np.einsum("xab, yca, bc -> xy",
                        st_out, st_in, self.op,
                        optimize=True)
        out -= np.einsum("xab, ybc, ca -> xy",
                         st_out, st_in, self.op,
                         optimize=True)
        return -1j * out

        # taking the real part implements the two parts of the commutator
        # return result.imag


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

        '''

        result = np.einsum("xab, bc, ycd, ad -> xy", st_out, self.op, st_in,
                           self.op.conj(), optimize=True)
        result -= 0.5 * np.einsum("xab, cb, cd, yda -> xy", st_out,
                                  self.op.conj(), self.op, st_in,
                                  optimize=True)

        result -= 0.5 * np.einsum("xab, ybc, dc, da -> xy", st_out, st_in,
                                  self.op.conj(), self.op, optimize=True)

        return result.real
        '''
        out = np.einsum("xab, bc, ycd, ad -> xy",
                        st_out, self.op,
                        st_in, self.op.conj(),
                        optimize=True)
        out -= 0.5*np.einsum("xab, cb, cd, yda -> xy",
                             st_out, self.op.conj(),
                             self.op, st_in,
                             optimize=True)
        out -= 0.5*np.einsum("xab, ybc, dc, da -> xy",
                             st_out, st_in,
                             self.op.conj(), self.op, optimize=True)
        return out


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
        u = np.diag([1, 1, 1, np.exp(1j*angle)]).reshape(2, 2, 2, 2)
        super().__init__(u)


class TwoPTMExplicit(TwoPTM):
    def __init__(self, ptm, bases_in, bases_out=None):
        assert len(bases_in) == 2
        if bases_out is None:
            bases_out = bases_in
        else:
            assert len(bases_out) == 2

        self._dim_hilbert = bases_in[0].dim_hilbert
        self._bases_in = bases_in
        self._bases_out = bases_out

        for b in chain(bases_in, bases_out):
            if b.dim_hilbert != self._dim_hilbert:
                raise ValueError(
                    'All bases must have the same Hilbert dimensionality.')
        self._validate_bases(bases_out=bases_out)

        assert ptm.shape == tuple(
            b.dim_pauli for b in chain(bases_out, bases_in))

        self._ptm = ptm

    @staticmethod
    def _combine_bases_vectors(bases):
        return reduce(np.kron, [b.basisvectors for b in bases])

    def _validate_bases(self, **kwargs):
        for name, bases in kwargs.items():
            if not isinstance(bases, tuple):
                raise ValueError(
                    "`{n}` should be a tuple, got {t}."
                    .format(n=name, t=type(bases)))
            for b in bases:
                if self._dim_hilbert != b.dim_hilbert:
                    raise ValueError(
                        "Expected bases with Hilbert dimensionality {}, "
                        "but {} has elements with Hilbert dimensionality {}."
                        .format(self._dim_hilbert, name, b.dim_hilbert))

    def get_matrix(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in

        if (bases_in == self._bases_in and bases_out == self._bases_out):
            return self._ptm

        shape = tuple(b.dim_pauli for b in chain(bases_out, bases_in))
        return np.einsum("xij, yji, yz, zkl, wlk -> xw",
                         self._combine_bases_vectors(bases_out),
                         self._combine_bases_vectors(self._bases_out),
                         self._ptm.reshape(
                             self._dim_hilbert**4, self._dim_hilbert**4),
                         self._combine_bases_vectors(self._bases_in),
                         self._combine_bases_vectors(bases_in),
                         optimize=True).real.reshape(shape)


class CompilerBlock:
    def __init__(
            self,
            bits,
            op,
            index=None,
            bitmap=None,
            cond=None,
            in_basis=None,
            out_basis=None):
        self.bits = bits
        self.op = op
        self.bitmap = bitmap
        self.cond = cond
        self.in_basis = in_basis
        self.out_basis = out_basis
        self.ptm = None

    def __repr__(self):
        if self.op == "measure" or self.op == "getdiag" or self.op == 'cond_op':
            opstring = self.op
        else:
            opstring = "ptm"

        bitsstring = ",".join(self.bits)

        basis_string = ""
        if self.in_basis:
            basis_string += "in:" + \
                "|".join(" ".join(b.basisvector_names) for b in self.in_basis)
        if self.out_basis:
            basis_string += " out:" + \
                "|".join(" ".join(b.basisvector_names) for b in self.out_basis)

        return "<{}: {} on {} {}>".format(
            self.__class__.__name__, opstring, bitsstring, basis_string)


class Operation:
    def __init__(self, bits, op, *, cond=None):
        self.bits = bits
        self.operator = op
        self.cond = cond


class TwoPTMCompiler:
    def __init__(self, operations, initial_bases=None):
        """Precompiles and optimizes a set of PTM applications for calculation.
        Compilation includes:

        - Contracting single-ptms into adjacent two-qubit ptms
        - choosing basis that facilitate partial tracing:
            - traced qubits in gellmann basis
            - not-traced qubits in general basis
        - examining sparsity and using truncated bases if applicable

        Parameters
        ----------
        operations : list of tuples
            Possible operations are:

            - `([bit], PTM)` for single PTM applications.
            - `([bit0, bit1], TwoPTM)` for two-qubit ptm applications.
            - `([bits], "measure")` for requesting partial trace of all but
              `bits`.

            bit names can be anything hashable.

        initial_bases : None or dict
            If `initial_basis` is None, the initial state is assumed fully
            separable.  Otherwise, its a dict, mapping bits to bases.
        """

        self.operations = operations

        self.bits = set()

        for op in self.operations:
            for b in op.bits:
                self.bits.add(b)

        self.initial_bases = initial_bases
        if self.initial_bases is None:
            self.initial_bases = []

        self.blocks = None
        self.compiled_blocks = None

    def contract_to_two_ptms(self):

        ctr = 0

        blocks = []
        active_block_idx = {}
        bits_in_block = {}

        for op in self.operations:

            for b in op.bits:
                if b not in active_block_idx:
                    new_bl = []
                    active_block_idx[b] = len(blocks)
                    blocks.append(new_bl)
                    bits_in_block[b] = [b]

            ctr += 1
            if op.operator == "measure" or op.operator == "getdiag":
                # measurement goes in single block
                measure_block = [(op.bits, op.operator, ctr)]
                blocks.append(measure_block)
                for b in op.bits:
                    del active_block_idx[b]
            elif op.operator == 'cond_op':
                cond_op_block = [(op.bits, op.operator, ctr, op.cond)]
                blocks.append(cond_op_block)
                for b in op.bits:
                    del active_block_idx[b]
            elif len(op.bits) == 1:
                blocks[active_block_idx[op.bits[0]]].append((op.bits, op, ctr))
            elif len(op.bits) == 2:
                b0, b1 = op.bits
                bl_i0, bl_i1 = active_block_idx[b0], active_block_idx[b1]
                if bl_i0 == bl_i1:
                    # qubits are in same block
                    blocks[bl_i0].append((op.bits, op, ctr))
                else:
                    if len(bits_in_block[b0]) == 2:
                        # b0 was in block with someone else, new block for b0
                        new_bl0 = []
                        bl_i0 = active_block_idx[b0] = len(blocks)
                        blocks.append(new_bl0)
                        bits_in_block[b0] = [b0]
                    if len(bits_in_block[b1]) == 2:
                        # b1 was in block with someone else, new block for b1
                        new_bl1 = []
                        bl_i1 = active_block_idx[b1] = len(blocks)
                        blocks.append(new_bl1)
                        bits_in_block[b1] = [b1]
                    # now we can be sure that both qb are in single block. We
                    # combine them:
                    bits_in_block[b0] = [b0, b1]
                    bits_in_block[b1] = [b0, b1]
                    # append earlier block to later block
                    if bl_i0 < bl_i1:
                        blocks[bl_i1].extend(blocks[bl_i0])
                        blocks[bl_i0] = []
                        blocks[bl_i1].append((op.bits, op, ctr))
                        active_block_idx[b0] = active_block_idx[b1]
                    else:
                        blocks[bl_i0].extend(blocks[bl_i1])
                        blocks[bl_i1] = []
                        blocks[bl_i0].append((op.bits, op, ctr))
                        active_block_idx[b1] = active_block_idx[b0]

        # active blocks move to end
        for b, bli in active_block_idx.items():
            bl = blocks[bli]
            blocks[bli] = []
            blocks.append(bl)

        self.blocks = [bl for bl in blocks if bl]
        self.abi = active_block_idx

    def make_block_ptms(self):
        self.compiled_blocks = []

        for bl in self.blocks:
            if bl[0][1] == "measure" or bl[0][1] == "getdiag":
                mbl = CompilerBlock(bits=bl[0][0], op=bl[0][1])
                self.compiled_blocks.append(mbl)
            elif bl[0][1] == 'cond_op':
                mbl = CompilerBlock(bits=bl[0][0], op=bl[0][1], cond=bl[0][3])
                self.compiled_blocks.append(mbl)
            else:
                product = TwoPTMProduct([])
                bit_map = {}
                for bits, op, i in bl:
                    if len(bits) == 1:
                        b, = bits
                        if b not in bit_map:
                            bit_map[b] = len(bit_map)
                        product.elements.append(([bit_map[b]], op))
                    if len(bits) == 2:
                        b0, b1 = bits
                        if b0 not in bit_map:
                            bit_map[b0] = len(bit_map)
                        if b1 not in bit_map:
                            bit_map[b1] = len(bit_map)
                        product.elements.append(
                            ([bit_map[b0], bit_map[b1]], op))

                # order the bitlist
                bits = list(bit_map.keys())
                if bit_map[bits[0]] == 1:
                    bits = list(reversed(bits))

                ptm_block = CompilerBlock(
                    bits=bits,
                    bitmap=bit_map,
                    op=product)
                self.compiled_blocks.append(ptm_block)

    def basis_choice(self, tol=1e-16):

        # for each block
        #   find previous blocks for involved qubits:
        #   If none: set in_basis from_initial basis
        #   else: Set in_basis from that block
        #
        #   find out_bases from sparsity analysis

        for i, cb in enumerate(self.compiled_blocks):
            cb.in_basis = []
            cb.out_basis = []
            for bit in cb.bits:
                previous = [cb2 for j, cb2 in enumerate(
                    self.compiled_blocks[:i]) if bit in cb2.bits]
                if len(previous) == 0:
                    # we are the first, use init_basis
                    if bit in self.initial_bases:
                        in_basis = self.initial_bases[bit]
                    else:
                        in_basis = GeneralBasis(cb.op.dim_hilbert)
                else:
                    previous = previous[-1]
                    bit_idx_in_previous = previous.bits.index(bit)
                    in_basis = previous.out_basis[bit_idx_in_previous]

                cb.in_basis.append(in_basis)

            # find out_basis and ptm matrix
            if cb.op == "measure":
                cb.out_basis = [b.get_classical_subbasis()
                                for b in cb.in_basis]
            elif cb.op == "getdiag" or cb.op == 'cond_op':
                cb.out_basis = cb.in_basis
            elif len(cb.in_basis) == 2:
                full_basis = [b.get_superbasis() for b in cb.in_basis]
                full_mat = cb.op.get_matrix(cb.in_basis, full_basis)
                sparse_out_0 = np.nonzero(np.einsum(
                    "abcd -> a", full_mat**2, optimize=True) > tol)[0]
                sparse_out_1 = np.nonzero(np.einsum(
                    "abcd -> b", full_mat**2, optimize=True) > tol)[0]
                cb.out_basis = [
                    full_basis[0].get_subbasis(sparse_out_0),
                    full_basis[1].get_subbasis(sparse_out_1)
                ]
                cb.ptm = cb.op.get_matrix(cb.in_basis, cb.out_basis)

    def run(self):
        if self.blocks is None:
            self.make_block_ptms()
        if self.compiled_blocks is None:
            self.contract_to_two_ptms()
            self.basis_choice()

# TODO:
# * better structure of outer products beyond two qubits
# * Best way is to have processes with types attached to them

# * more explicit support for PTMs that are dimension-agnostic
# * more reasonable names (SuperOperator, Process, DiffProcess or so)
# * Singletons/caches to prevent recalculation
# * smarter handling of product intermediate basis
#   * domain and image hints
#   * automatic sparsification
# * qutip interfacing for me_solve
# * using auto-forward-differentiation to integrate processes?
# * return matric reps in other forms (process matrix, chi matrix?)
# * PTM compilation using circuit interface!
