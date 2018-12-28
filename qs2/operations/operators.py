import abc
from functools import reduce
import numpy as np


ERR_MSGS = dict(
    basis_dim_mismatch='The dimensions of the given basis do not match the provided operators: operator shape is {}, while basis has dimensions {}',
    not_sqr='Provided matrices are not square: provided matrix shape is {}',
    wrong_dim='Incorred dimensionality of the provided operator: operator dimensions are: {}',
    pauli_not_sqr='The Pauli dimension is not an exact square of some HIlbert dimension, got Pauli dimension of {}'
)


class Operator(metaclass=abc.ABCMeta):
    def __init__(self, matrix, num_subspaces):
        self._matrix = matrix
        self._num_subspaces = num_subspaces

    @property
    def matrix(self):
        return self._matrix

    @property
    def num_subspaces(self):
        return self._num_subspaces

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
        super().__init__(matrix, len(bases))
        self._bases = bases

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
    def dim_hilbert(self):
        return np.sqrt(self._matrix.shape[0])

    @property
    def dim_pauli(self):
        return self._matrix.shape[0]

    def to_ptm(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in

        if bases_in == self._bases and bases_out == self._bases:
            return self

        self._check_consistent_basis(self._matrix, bases_in)
        self._check_consistent_basis(self._matrix, bases_out)

        cur_vectors = [basis.vectors for basis in self._bases]
        cur_tensor = reduce(np.kron, cur_vectors)

        in_vectors = [basis.vectors for basis in bases_in]
        in_tensor = reduce(np.kron, in_vectors)

        if bases_out == bases_in:
            out_tensor = in_tensor
        else:
            out_vectors = [basis.vectors for basis in bases_out]
            out_tensor = reduce(np.kron, out_vectors)

        ptm = np.einsum("xij, yji, yz, zkl, wlk -> xw", out_tensor,
                        cur_tensor, self._matrix, cur_tensor, in_tensor, optimize=True).real

        return PTMOperator(ptm, bases_out)


class KrausOperator(Operator):
    def __init__(self, matrix, num_subspaces):
        assert isinstance(matrix, np.ndarray)

        if matrix.ndim == 2:
            matrix = np.array([matrix])

        self._check_matrix(matrix)
        super().__init__(matrix, num_subspaces)

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

    @property
    def dim_hilbert(self):
        return self._matrix.shape[1]

    @property
    def dim_pauli(self):
        dim_hilbert = self._matrix.shape[1]
        return dim_hilbert * dim_hilbert

    def to_ptm(self, bases_in, bases_out=None):
        if bases_out is None:
            bases_out = bases_in

        in_vectors = [basis.vectors for basis in bases_in]
        in_tensor = reduce(np.kron, in_vectors)

        if bases_out == bases_in:
            out_tensor = in_tensor
        else:
            out_vectors = [basis.vectors for basis in bases_out]
            out_tensor = reduce(np.kron, out_vectors)

        ptm = np.einsum("xab, zbc, ycd, zad -> xy", out_tensor, self._matrix,
                        in_tensor, self._matrix.conj(), optimize=True).real

        return PTMOperator(ptm, bases_out)
