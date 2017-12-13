import numpy as np
import collections

import scipy.linalg


"The transformation matrix between the two bases. Its essentially a Hadamard, so its its own inverse."
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
    """
    The vector of 'Pauli matrices' in dimension n.
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
    """Transform a Pauli transfer in the "usual" basis (0xyz) [1],
    to the 0xy1 basis which is required by sparsesdm.apply_ptm.

    If general_basis is True, transform to the 01xy basis, which is the
    two-qubit version of the general basis defined by ptm.general_ptm_basis_vector().

    ptm: The input transfer matrix in 0xyz basis. Can be 4x4, 4x3 or 3x3 matrix of real numbers.

         If 4x4, the first row must be (1,0,0,0). If 4x3, this row is considered to be omitted.
         If 3x3, the transformation is assumed to be unitary, thus it is assumed that
         the first column is also (1,0,0,0) and was omitted.

    [1] Daniel Greenbaum, Introduction to Quantum Gate Set Tomography, http://arxiv.org/abs/1509.02921v1
    """

    ptm = np.array(ptm)

    if ptm.shape == (3, 3):
        ptm = np.hstack(([[0], [0], [0]], ptm))

    if ptm.shape == (3, 4):
        ptm = np.vstack(([1, 0, 0, 0], ptm))

    assert ptm.shape == (4, 4)
    assert np.allclose(ptm[0, :], [1, 0, 0, 0])

    # result = np.dot(
    # basis_transformation_matrix, np.dot(
    # ptm, basis_transformation_matrix))

    if general_basis:
        result = ExplicitBasisPTM(
            ptm, PauliBasis_exyz()).get_matrix(
            GeneralBasis(2))
    else:
        result = ExplicitBasisPTM(
            ptm, PauliBasis_exyz()).get_matrix(
            PauliBasis_0xy1())

    return result


def to_0xyz_basis(ptm):
    """Transform a Pauli transfer in the 0xy1 basis [1],
    to the the usual 0xyz. The inverse of to_0xy1_basis.

    ptm: The input transfer matrix in 0xy1 basis. Must be 4x4.

    [1] Daniel Greenbaum, Introduction to Quantum Gate Set Tomography, http://arxiv.org/abs/1509.02921v1
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
            "Dimensions wrong, must be one- or two Pauli transfer matrix ")


def hadamard_ptm(general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing perfect unitary Hadamard (Rotation around the (x+z)/sqrt(2) axis by π).
    """
    u = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
    if general_basis:
        pb = GeneralBasis(2)
    else:
        pb = PauliBasis_0xy1()
    return ConjunctionPTM(u).get_matrix(pb)


def amp_ph_damping_ptm(gamma, lamda, general_basis=False):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing amplitude and phase damping with parameters gamma and lambda.
    (See Nielsen & Chuang for definition.)
    """

    if general_basis:
        pb = GeneralBasis(2)
    else:
        pb = PauliBasis_0xy1()

    return AmplitudePhaseDampingPTM(gamma, lamda).get_matrix(pb)

def gen_amp_damping_ptm(gamma_down, gamma_up):
    """Return a 4x4 Pauli transfer matrix  representing amplitude damping including an excitation rate gamma_up.
    """

    gamma = gamma_up + gamma_down
    p = gamma_down / (gamma_down + gamma_up)

    ptm = np.array([
        [1, 0, 0, 0],
        [0, np.sqrt((1 - gamma)), 0, 0],
        [0, 0, np.sqrt((1 - gamma)), 0],
        [(2 * p - 1) * gamma, 0, 0, 1 - gamma]]
    )

    return to_0xy1_basis(ptm)


def dephasing_ptm(px, py, pz):
    """Return a 4x4 Pauli transfer matrix in 0xy1 basis,
    representing dephasing (shrinking of the Bloch sphere along the principal axes),
    with different rates across the different axes.
    p_i/2 is the flip probability, so p_i = 0 corresponds to no shrinking, while p_i = 1 is total dephasing.
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
    assert kraus.shape == (d, d)

    st = general_ptm_basis_vector(d)

    return np.einsum("xab, bc, ycd, ad -> xy", st,
                     kraus, st, kraus.conj()).real


def single_kraus_to_ptm(kraus, general_basis=False):
    """Given a Kraus operator in z-basis, obtain the corresponding single-qubit ptm in 0xy1 basis"""
    if general_basis:
        st = general_ptm_basis_vector(2)
    else:
        st = single_tensor
    return np.einsum("xab, bc, ycd, ad -> xy", st,
                     kraus, st, kraus.conj()).real


def double_kraus_to_ptm(kraus, general_basis=False):
    if general_basis:
        st = general_ptm_basis_vector(2)
    else:
        st = single_tensor

    dt = np.kron(st, st)

    return np.einsum("xab, bc, ycd, ad -> xy", dt,
                     kraus, dt, kraus.conj()).real


class PauliBasis:
    def __init__(self, basisvectors=None):
        """
        Defines a Pauli basis [1]. The number of element vectors is given by `dim_pauli`,
        while the dimension of the hilbert space is given by dim_hilbert.

        For instance, for a qubit (Hilbert space dimension d=2), one could employ a Pauli basis
        with dimension d**2 = 4, or, if one wants do describe a classical state (mixture of |0> and |1>),
        use a smaller basis with only d_pauli = 2.

        [1] A Pauli basis is an orthonormal basis (w.r.t <A, B> = Tr(A.B+)) for a space of Hermitian matrices.
        """

        "a tensor B of shape (dim_pauli, dim_hilbert, dim_hilbert)"
        "read as a vector of matrices"
        "must satisfy Tr(B[i] @ B[j]) = delta(i, j)"
        if basisvectors is not None:
            self.basisvectors = basisvectors

        shape = self.basisvectors.shape

        assert shape[1] == shape[2]

        self.dim_hilbert = shape[1]
        self.dim_pauli = shape[0]

        self.computational_basis_vectors = np.einsum(
            "xii -> ix", self.basisvectors)

    def hilbert_to_pauli_vector(self, rho):
        return np.einsum("xab, ba -> x", self.basisvectors, rho)


class GeneralBasis(PauliBasis):
    def __init__(self, dim):
        self.basisvectors = general_ptm_basis_vector(dim)
        super().__init__()


class PauliBasis_0xy1(PauliBasis):
    "the pauli basis used by older versions of quantumsim"
    basisvectors = np.array([[[1, 0], [0, 0]],
                             np.sqrt(0.5) * np.array([[0, 1], [1, 0]]),
                             np.sqrt(0.5) * np.array([[0, -1j], [1j, 0]]),
                             [[0, 0], [0, 1]]])


class PauliBasis_exyz(PauliBasis):
    "standard Pauli basis for a qubit"
    basisvectors = np.sqrt(0.5) * np.array([[[1, 0], [0, 1]],
                                            [[0, 1], [1, 0]],
                                            [[0, -1j], [1j, 0]],
                                            [[1, 0], [0, -1]]])


class PTM:
    def __init__(self):
        """
        A Pauli transfer matrix. ABC
        """

        "the hilbert space dimension on which the PTM operates"
        self.dim_hilbert = ()
        raise NotImplementedError

    def get_matrix(self, in_basis, out_basis=None):
        """
        Return the matrix representation of this PTM in the basis given.

        If out_basis is None, in_basis = out_basis is assumed.

        If out_basis spans only a subspace, projection on that subspace is implicit.
        """
        raise NotImplementedError

    def __add__(self, other):
        assert isinstance(other, PTM), "cannot add PTM to non-ptm"
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


class ExplicitBasisPTM(PTM):
    def __init__(self, ptm, basis):
        self.ptm = ptm
        self.basis = basis

        assert self.ptm.shape == (self.basis.dim_pauli, self.basis.dim_pauli)

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        result = np.einsum("xab, yba, yz, zcd, wdc -> xw",
                           basis_out.basisvectors,
                           self.basis.basisvectors,
                           self.ptm,
                           self.basis.basisvectors,
                           basis_in.basisvectors).real

        return result


class LinearCombPTM(PTM):
    def __init__(self, elements):
        """
        A linear combination of other PTMs.
        Should usually not be instantiated by hand, but created as a result of adding or scaling PTMs.
        """

        # check dimensions
        dimensions_set = set(p.dim_hilbert for p in elements.keys())
        if len(dimensions_set) > 1:
            raise ValueError(
                "cannot create linear combination: incompatible dimensions: {}".format(dimensions_set))

        self.dim_hilbert = dimensions_set.pop()

        # float defaultdict of shape {ptm: coefficient, ...}
        self.elements = collections.Counter(elements)

    def get_matrix(self, in_basis, out_basis=None):
        return sum(c * p.get_matrix(in_basis, out_basis)
                   for p, c in self.elements.items())

    def __mul__(self, scalar):
        return LinearCombPTM({p: scalar * c for p, c in self.elements.items()})

    def __add__(self, other):
        assert isinstance(other, PTM), "cannot add PTM to non-ptm"
        if isinstance(other, LinearCombPTM):
            return LinearCombPTM(self.elements + other.elements)
        else:
            return LinearCombPTM(self.elements + collections.Counter([other]))


class ProductPTM(PTM):
    def __init__(self, elements):
        """
        A product of other PTMs.
        Should usually not be instantiated by hand, but created as a result of multiplying PTMs.

        elements: list of factors. Will be multiplied in order elements[n-1] @ ... @ elements[0].
        """

        # check dimensions
        dimensions_set = set(p.dim_hilbert for p in elements)
        if len(dimensions_set) > 1:
            raise ValueError(
                "cannot create product: incompatible dimensions: {}".format(dimensions_set))

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

        assert basis_in.dim_hilbert == basis_out.dim_hilbert

        if self.dim_hilbert:
            assert basis_in.dim_hilbert == self.dim_hilbert

        complete_basis = GeneralBasis(basis_in.dim_hilbert)
        result = np.eye(complete_basis.dim_pauli)
        for pi in self.elements:
            pi_mat = pi.get_matrix(complete_basis)
            result = result @ pi_mat

        trans_mat_in = np.einsum(
            "xab, yba",
            complete_basis.basisvectors,
            basis_in.basisvectors)
        trans_mat_out = np.einsum(
            "xab, yba",
            basis_out.basisvectors,
            complete_basis.basisvectors)

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
        """
        The PTM describing conjunction with unitary operator `op`, i.e.

        rho' = P(rho) = op . rho . op^dagger

        `op` is a matrix given in computational basis.

        Typical usage: op describes the unitary time evolution of a system described by rho.
        """
        self.op = np.array(op)
        self.dim_hilbert = self.op.shape[0]

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        assert self.op.shape == (basis_out.dim_hilbert, basis_in.dim_hilbert)

        st_out = basis_out.basisvectors
        st_in = basis_in.basisvectors

        result = np.einsum("xab, bc, ycd, ad -> xy",
                           st_out, self.op, st_in, self.op.conj())

        assert np.allclose(result.imag, 0)

        return result.real


class IntegratedPLM(PTM):
    def __init__(self, plm):
        """
        The PTM that arises for applying the Pauli Liouvillian `plm`
        for one unit of time.
        """
        self.plm = plm
        self.dim_hilbert = plm.dim_hilbert

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        assert self.op.shape == (basis_out.dim_hilbert, basis_in.dim_hilbert)

        # we need to get a square representation!
        plm_matrix = self.plm.get_matrix(basis_in, basis_in)

        ptm_matrix = scipy.linalg.matfuncs.expm(plm_matrix)

        # then basis-transform to out basis
        return PTM(ptm_matrix).get_matrix()


class AdjunctionPLM(PTM):
    def __init__(self, op):
        """
        The PLM (Pauli Liouvillian Matrix) describing adjunction with operator `op`, i.e.

        rho' = P(rho) =  1j*(op.rho - rho.op)

        Typical usage: op is a Hamiltonian, the PTM describes the infinitesimal evolution of rho.

        """
        self.op = op
        self.dim_hilbert = op.shape[0]

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        assert self.op.shape == (basis_out.dim_hilbert, basis_in.dim_hilbert)

        st_out = basis_out.basisvectors
        st_in = basis_in.basisvectors

        result = 1j * np.einsum("xab, bc, ycd -> xy",
                                st_out, self.op, st_in)

        # taking the real part implements the two parts of the commutator
        return result.real


class LindbladPLM(PTM):
    def __init__(self, op):
        """
        The PLM describing the Lindblad superoperator of `op`, i.e.

        rho' = P(rho) = op.rho.op^dagger - 1/2 {op^dagger.op, rho}

        Typical usage: op is a decay operator.
        """
        self.op = op
        self.dim_hilbert = op.shape[0]

    def get_matrix(self, basis_in, basis_out=None):
        if basis_out is None:
            basis_out = basis_in

        assert self.op.shape == (basis_out.dim_hilbert, basis_in.dim_hilbert)

        st_out = basis_out.basisvectors
        st_in = basis_in.basisvectors

        result = np.einsum("xab, bc, ycd, ad -> xy",
                           st_out, self.op, st_in, self.op.conj())

        result -= 0.5 * np.einsum("xab, cb, cd, yda -> xy",
                                  st_out, self.op.conj(), self.op, st_in)

        result -= 0.5 * np.einsum("xab, ybc, dc, da -> xy",
                                  st_out, st_in, self.op.conj(), self.op)

        return result.real


class RotateXPTM(ConjunctionPTM):
    def __init__(self, angle):
        s, c = np.sin(angle / 2), np.cos(angle / 2)
        super().__init__([[c, -1j * s], [-1j * s, c]])


class RotateYPTM(ConjunctionPTM):
    def __init__(self, angle):
        s, c = np.sin(angle / 2), np.cos(angle / 2)
        super().__init__([[c, s], [-s, c]])


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


# TODO:
# * thought + test on how to handle multi-qubit ptms
# * more explicit support for PTMs that are dimension-agnostic
# * more reasonable names (SuperOperator, Process, DiffProcess or so)
# * Singletons/caches to prevent recalculation
# * smarter handling of product intermediate basis
#   * domain and image hints
#   * automatic sparsification
# * qutip interfacing for me_solve
# * using auto-forward-differentiation to integrate processes?
# * return matric reps in other forms (process matrix, chi matrix?)
# * PTM compilation using circuit interface?
# * Basis vector names
