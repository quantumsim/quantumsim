import abc
from functools import reduce
from functools import lru_cache, wraps
from hashlib import sha1
import numpy as np


ERR_MSGS = dict(
    basis_dim_mismatch='The dimensions of the given basis do not match the provided operators: operator shape is {}, while basis has dimensions {}',
    not_sqr='Provided matrices are not square: provided matrix shape is {}',
    wrong_dim='Incorred dimensionality of the provided operator: operator dimensions are: {}',
    pauli_not_sqr='The Pauli dimension is not an exact square of some HIlbert dimension, got Pauli dimension of {}'
)


class Operator(metaclass=abc.ABCMeta):
    '''A general operator which effectively implements a quantum process. In general an operator is only identified by it's matrix. However the Operator class contains additionaly information about the type of operator and additional data that goes along with it. For example for the case of a Pauli Transfer Matrix (PTM) operator in addition to the array representing the PTM itself the bases in which the PTM has been expanded in are also bundeled. These bases are then used for operator transformation and operator properties.

    Additionally each operator is associated with methods, which check the correctness of the input parameters (with respect to the specific operator type). Finally as operations in quantumsim are ultimately performed with the PTM repesentation, each operator has a method which compiles the equivalent PTM repesentation in a specified basis.

    Currently the supported types of operators include: Kraus, Pauli Transfer Matrix and Unitary operators.
    '''

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
        """Checks that the dimesnions of the matrix and the shape are as expected

        Parameters
        ----------
        matrix : np.ndarray
            The matrix that is the mathematical description of the operator

        """

        pass

    @abc.abstractmethod
    def _check_consistent_basis(self, matrix, bases):
        pass

    @abc.abstractmethod
    def to_ptm(self, bases_in, bases_out=None):
        """Transforms the current type of Operator to a PTM operator expanded in explicit bases.

        Parameters
        ----------
        bases_in : tuple
            The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator
        bases_out : tuple, optional
            The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator (the default is None, which means that bases_out are the same as bases_in)

        """

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
        """A Krause operator, which is mathematically expressed as a set of kraus matrices, representing the decomposition of the unitary operator (or other operator types such as choi, ptm etc). Additionally the hilbert dimensions of the individual operator subspaces need to be provided as a tuple, for the purpose of consistency and analysis (of basis, operator compatibility when joining, etc...)

        Parameters
        ----------
        matrix : np.ndarray
            The set of kraus matrices repesenting the process
        subspace_dim_hilbert : tuple
            The tuple of the individual hilbert dimensions of each subspace

        """

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

        ptm = kraus_to_ptm(self.matrix, bases_in, bases_out)

        return PTMOperator(ptm, bases_out)


class UnitaryOperator(Operator):
    def __init__(self, matrix, subspace_dim_hilbert):
        """A bit of a misleading name, but it is a general unitary operator (most likely in the pauli (gell_mann) basis). This should actually be a subtype of the Kraus operator, but I am not sure.
        """

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


def hashed_lru_cache(function):
    """Wrapper function that is used to cache operator conversions data. Since lists and arrays are not by default cached by lru_cache and can in general be somewhat large, the arrays are instead hashed using the sha1 protocol. This choice is motivated by the speed of execution of the sha1 hashing and the correct handling of the data types of the arrays.

    Once the hash is obtained, the lru_cache wrapped function is called with the hashed repesentation of the array. If that repsentation is in the cache, the corresponding result is returned. If not, then the array from which the hash was obtained is used to calculate the transformation.
    """

    cur_arr = None

    @lru_cache(maxsize=64)
    def cached_wrapper(_, *args, **kwargs):
        nonlocal cur_arr
        return function(cur_arr, *args, **kwargs)

    @wraps(function)
    def wrapper(array, *args, **kwargs):
        nonlocal cur_arr
        # sha1 is faster than converting array to tuple or
        # hashing a string. Alternatively one could do:
        #arr_hash = hash(np.array_str(array.copy(order='C')))
        # this requieres arrays to be C-contigious.
        arr_hash = sha1(array.copy(order='C')).hexdigest()
        cur_arr = array
        return cached_wrapper(arr_hash, *args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


@hashed_lru_cache
def change_ptm_basis(cur_ptm, cur_bases, bases_in, bases_out):
    """Function to convert a PTM matrix expanded in a certain basis, to an equivalent PTM expanded in a different basis, specified by bases_in, bases_out

    Parameters
    ----------
    cur_ptm : np.ndarray
        The PTM matrix expanded in the current basis
    cur_bases : tuple
        The tuple of the qs2.Pauli_Basis objects corresponding the current basis of each subspace of the operator that the PTM was expanded in
    bases_in : tuple
        The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator
    bases_out : tuple
        The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator

    Returns
    -------
    np.ndarray
        The PTM matrix expanded in the provided basis
    """
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


@hashed_lru_cache
def kraus_to_ptm(kraus, bases_in, bases_out):
    """Function to convert a set of kraus matrices to a PTM matrix expanded in a basis, specified by bases_in, bases_out

    Parameters
    ----------
    kraus : np.ndarray
        The set of kraus matrices
    bases_in : tuple
        The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator
    bases_out : tuple
        The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator

    Returns
    -------
    np.ndarray
        The PTM matrix expanded in the provided basis
    """

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


@hashed_lru_cache
def unitary_to_ptm(unitary, bases_in, bases_out):
    """Function to convert a unitary matrix to a PTM matrix expanded in a basis, specified by bases_in, bases_out

    Parameters
    ----------
    unitary : np.ndarray
        The unitary matrix
    bases_in : tuple
        The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator
    bases_out : tuple
        The tuple of the qs2.Pauli_Basis object corresponding to each subspace of the operator

    Returns
    -------
    np.ndarray
        The PTM matrix expanded in the provided basis
    """
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


@hashed_lru_cache
def lindblad_to_ptm(lindbladian, bases_in, bases_out):

    in_vectors = [basis.vectors for basis in bases_in]
    in_tensor = reduce(np.kron, in_vectors)

    if bases_out == bases_in:
        out_tensor = in_tensor
    else:
        out_vectors = [basis.vectors for basis in bases_out]
        out_tensor = reduce(np.kron, out_vectors)

    ptm = np.einsum("xab, bc, ycd, ad -> xy",
                    out_tensor, lindbladian,
                    in_tensor, lindbladian.conj(),
                    optimize=True)

    ptm -= 0.5 * np.einsum("xab, cd, yda, cb  -> xy",
                           out_tensor, lindbladian, in_tensor, lindbladian.conj(),
                           optimize=True)

    ptm -= 0.5 * np.einsum("xab, da, ybc, dc  -> xy",
                           out_tensor, lindbladian,
                           in_tensor, lindbladian.conj(),
                           optimize=True)

    return ptm.real


@hashed_lru_cache
def adjunction_to_ptm(lindbladian, bases_in, bases_out):
    in_vectors = [basis.vectors for basis in bases_in]
    in_tensor = reduce(np.kron, in_vectors)

    if bases_out == bases_in:
        out_tensor = in_tensor
    else:
        out_vectors = [basis.vectors for basis in bases_out]
        out_tensor = reduce(np.kron, out_vectors)

    ptm = np.einsum("xab, bc, yca -> xy",
                    out_tensor, lindbladian,
                    in_tensor, optimize=True)

    return ptm.imag
