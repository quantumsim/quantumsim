import abc
from functools import reduce
from functools import lru_cache
import numpy as np


ERR_MSGS = dict(
    basis_dim_mismatch='The dimensions of the given basis do not match the provided operators: operator shape is {}, while basis has dimensions {}',
    not_sqr='Provided matrices are not square: provided matrix shape is {}',
    wrong_dim='Incorred dimensionality of the provided operator: operator dimensions are: {}',
    pauli_not_sqr='The Pauli dimension is not an exact square of some HIlbert dimension, got Pauli dimension of {}'
)


class Operator(metaclass=abc.ABCMeta):
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    @abc.abstractmethod
    def num_subspaces(self):
        pass

    @property
    @abc.abstractmethod
    def size(self):
        pass

    @property
    @abc.abstractmethod
    def dim_hilbert(self):
        pass

    @property
    @abc.abstractmethod
    def dim_pauli(self):
        pass

    @abc.abstractmethod
    def _check_matrix(self, matrix):
        pass

    @abc.abstractmethod
    def _check_consistent_basis(self, matrix, bases):
        pass

    @abc.abstractmethod
    def to_ptm(self, bases_in, bases_out=None):
        pass


class PTMOperator(Operator):
    def __init__(self, matrix, bases):
        assert isinstance(matrix, np.ndarray)
        self._check_matrix(matrix)
        self._check_consistent_basis(matrix, bases)

        super().__init__(matrix)
        self.bases = bases

    def _check_matrix(self, matrix):
        if matrix.ndim != 2:
            raise ValueError(ERR_MSGS['wrong_dim'].format(matrix.shape))

        dim_pauli = matrix.shape[0]

        if matrix.shape != (dim_pauli, dim_pauli):
            raise ValueError(ERR_MSGS['not_sqr'].format(matrix.shape))

        dim_hilbert = int(np.sqrt(dim_pauli))
        if dim_pauli != dim_hilbert*dim_hilbert:
            raise ValueError(ERR_MSGS['pauli_not_sqr'].format(dim_pauli))

    def _check_consistent_basis(self, matrix, bases):
        mat_dim_pauli = matrix.shape[0]
        basis_dim_pauli = np.prod([basis.dim_pauli for basis in bases])

        if mat_dim_pauli != basis_dim_pauli:
            raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                mat_dim_pauli, basis_dim_pauli))

    @property
    def num_subspaces(self):
        return len(self.bases)

    @property
    def dim_hilbert(self):
        return tuple([basis.dim_hilbert for basis in self.bases])

    @property
    def size(self):
        return np.product(self.dim_hilbert) ** 2

    @property
    def dim_pauli(self):
        return tuple([basis.dim_pauli for basis in self.bases])

    def to_ptm(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in

        if bases_in == self.bases and bases_out == self.bases:
            return self

        self._check_consistent_basis(self.matrix, bases_in)
        self._check_consistent_basis(self.matrix, bases_out)

        ptm = change_ptm_basis(self.matrix, self.bases,
                               bases_in, bases_out)

        return PTMOperator(ptm, bases_out)


class KrausOperator(Operator):
    def __init__(self, matrix, subspace_dim_hilbert):
        assert isinstance(matrix, np.ndarray)

        if matrix.ndim == 2:
            matrix = np.array([matrix])

        self._check_matrix(matrix)
        self._check_subspace_dims(matrix, subspace_dim_hilbert)

        super().__init__(matrix)
        self._dim_hilbert = subspace_dim_hilbert
        self._dim_pauli = [sub_dim_hilbert * sub_dim_hilbert
                           for sub_dim_hilbert in self.dim_hilbert]

    def _check_matrix(self, matrix):
        if matrix.ndim < 3:
            raise ValueError(ERR_MSGS['wrong_dim'].format(matrix.shape))

        dim_hilbert = matrix.shape[1]

        if matrix.shape[1:] != (dim_hilbert, dim_hilbert):
            raise ValueError(ERR_MSGS['not_sqr'].format(matrix.shape))

    def _check_consistent_basis(self, matrix, bases):
        mat_dim_hilbert = matrix.shape[1]
        basis_dim_hilbert = np.prod([basis.dim_hilbert for basis in bases])

        if mat_dim_hilbert != basis_dim_hilbert:
            raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                mat_dim_hilbert, basis_dim_hilbert))

    def _check_subspace_dims(self, matrix, subspace_dim_hilbert):
        dim_hilbert = matrix.shape[1]
        if np.prod(subspace_dim_hilbert) != dim_hilbert:
            raise ValueError('Incorrect hilbert dimensions of the subspaces')

    @property
    def num_subspaces(self):
        return len(self.dim_hilbert)

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def dim_pauli(self):
        return self._dim_pauli

    @property
    def size(self):
        return np.product(self.dim_pauli)

    def to_ptm(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in

        self._check_consistent_basis(self.matrix, bases_in)
        self._check_consistent_basis(self.matrix, bases_out)

        mat = self.matrix
        print(mat)
        ptm = kraus_to_ptm(mat, bases_in, bases_out)

        return PTMOperator(ptm, bases_out)


class UnitaryOperator(Operator):
    def __init__(self, matrix, subspace_dim_hilbert):
        assert isinstance(matrix, np.ndarray)

        self._check_matrix(matrix)
        self._check_subspace_dims(matrix, subspace_dim_hilbert)

        super().__init__(matrix)
        self._dim_hilbert = subspace_dim_hilbert
        self._dim_pauli = [sub_dim_hilbert * sub_dim_hilbert
                           for sub_dim_hilbert in self.dim_hilbert]

    def _check_matrix(self, matrix):

        dim_hilbert = matrix.shape[0]

        if matrix.shape != (dim_hilbert, dim_hilbert):
            raise ValueError(ERR_MSGS['not_sqr'].format(matrix.shape))

    def _check_consistent_basis(self, matrix, bases):
        mat_dim_hilbert = matrix.shape[0]
        basis_dim_hilbert = np.prod([basis.dim_hilbert for basis in bases])

        if mat_dim_hilbert != basis_dim_hilbert:
            raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                mat_dim_hilbert, basis_dim_hilbert))

    def _check_subspace_dims(self, matrix, subspace_dim_hilbert):
        dim_hilbert = matrix.shape[0]
        if np.prod(subspace_dim_hilbert) != dim_hilbert:
            raise ValueError('Incorrect hilbert dimensions of the subspaces')

    @property
    def num_subspaces(self):
        return len(self.dim_hilbert)

    @property
    def dim_hilbert(self):
        return self._dim_hilbert

    @property
    def dim_pauli(self):
        return self._dim_pauli

    @property
    def size(self):
        return np.product(self.dim_hilbert)

    def to_ptm(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in

        self._check_consistent_basis(self.matrix, bases_in)
        self._check_consistent_basis(self.matrix, bases_out)

        ptm = unitary_to_ptm(self.matrix, bases_in, bases_out)

        return PTMOperator(ptm, bases_out)

    def embed_dim_hilbert(self, new_dim_hilbert, inds=None):
        if inds is None:
            ind_range = range(min(self.dim_hilbert, new_dim_hilbert))
            inds = np.array([(i, j) for i, j in enumerate(ind_range)])

        proj = np.zeros((self.dim_hilbert, new_dim_hilbert))
        proj[tuple(inds.T)] = 1

        new_unitary = np.eye(new_dim_hilbert) - \
            proj.T @ proj + proj.T @ self.matrix @ proj

        return UnitaryOperator(new_unitary, new_dim_hilbert)


def change_ptm_basis(cur_ptm, cur_bases, bases_in, bases_out):
    cur_vectors = [basis.vectors for basis in cur_bases]
    cur_tensor = reduce(np.kron, cur_vectors)

    in_vectors = [basis.vectors for basis in bases_in]
    in_tensor = reduce(np.kron, in_vectors)

    if bases_out == bases_in:
        out_tensor = in_tensor
    else:
        out_vectors = [basis.vectors for basis in bases_out]
        out_tensor = reduce(np.kron, out_vectors)

    ptm = np.einsum("xij, yji, yz, zkl, wlk -> xw",
                    out_tensor, cur_tensor,
                    cur_ptm,
                    cur_tensor, in_tensor,
                    optimize=True).real

    return ptm


def kraus_to_ptm(kraus: np.ndarray, bases_in, bases_out):
    in_vectors = [basis.vectors for basis in bases_in]
    in_tensor = reduce(np.kron, in_vectors)

    if bases_out == bases_in:
        out_tensor = in_tensor
    else:
        out_vectors = [basis.vectors for basis in bases_out]
        out_tensor = reduce(np.kron, out_vectors)

    ptm = np.einsum("xab, zbc, ycd, zad -> xy",
                    out_tensor, kraus,
                    in_tensor, kraus.conj(),
                    optimize=True).real

    return ptm


def unitary_to_ptm(unitary, bases_in, bases_out):
    in_vectors = [basis.vectors for basis in bases_in]
    in_tensor = reduce(np.kron, in_vectors)

    if bases_out == bases_in:
        out_tensor = in_tensor
    else:
        out_vectors = [basis.vectors for basis in bases_out]
        out_tensor = reduce(np.kron, out_vectors)

    ptm = np.einsum("xab, bc, ycd, ad -> xy",
                    out_tensor, unitary,
                    in_tensor, unitary.conj(),
                    optimize=True).real

    return ptm
