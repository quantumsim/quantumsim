import numpy as np
from .. import bases

ERR_MSGS = dict(
    basis_dim_mismatch='The dimensions of the given basis do not match the provided operators: operator shape is {}, while basis has dimensions {}',
    not_sqr='Provided matrices are not square: provided matrix shape is {}',
    wrong_dim='Incorred dimensionality of the provided operator: operator dimensions are: {}',
    pauli_not_sqr='The Pauli dimension is not an exact square of some HIlbert dimension, got Pauli dimension of {}'
)


def kraus_to_ptm(kraus, pauli_basis=None):
    """Converts a kraus matrix or an array of kraus matrices to the Pauli transfer matrix (PTM) representation.

    Parameters
    ----------
    kraus : ndarray
        Either the square kraus operator or an array of square kraus operators representing a process.
    pauli_basis : PauliBasis, optional
        The provided pauli basis in which the PTM is expanded (the default is None, which corrsponds to an automatically generated general Pauli basis)

    Raises
    ------
    ValueError
        If kraus is not a 2-dimensional or 3-dimensional matrix
    ValueError
        If the provided Kraus matrices are not all square
    ValueError
        If the provided PauliBasis' dimensions do not match the dimensions of the provided Kraus operators
    ValueError
        If the provided subspaces hilbert dimensions do not match the dimension of the system (as set by the Kraus operators)

    Returns
    -------
    ndarray
        The PTM representation of the process
    """

    kraus = _check_kraus_dims(kraus)
    dim = kraus.shape[1]

    # Generate the basis vectors of the full system
    if pauli_basis is not None:
        _check_kraus_basis_consistency(kraus, pauli_basis)
        vectors = pauli_basis.vectors
    else:
        vectors = bases.general(dim).vectors

    ptm = np.einsum("xab, zbc, ycd, zad -> xy", vectors, kraus,
                    vectors, kraus.conj(), optimize=True).real
    return ptm


def ptm_to_choi(ptm, pauli_basis=None):
    """Converts a Pauli transfer matrix representation (PTM) of a process to the corresponding choi matrix representation (Choi).

    Parameters
    ----------
    ptm : ndarray
        The PTM of the process
    pauli_basis : PauliBasis, optional
        The Pauli basis used for the representations (the default is None, which corrsponds to an automatically generated general Pauli basis)
    subs_dim_hilbert : ndarray, optional
        Array of the hilbert dimensionalities of each subsystem (the default is None, which corresponds to 1 subsystem with the same dimensionality as that of the Kraus operator)

    Raises
    ------
    ValueError
        If the PTM is not a 2-dimensional square matrix.
    ValueError
        If the provided PauliBasis' dimensions do not match the dimensions of the provided PTM
    ValueError
        If the provided subspaces' hilbert dimensions do not match the dimension of the system (as set by the PTM)

    Returns
    -------
    ndarray
        The Choi matrix representation of the process
    """
    _check_ptm_or_choi_dims(ptm)
    dim = ptm.shape[0]

    if pauli_basis is not None:
        _check_ptm_or_choi_basis_consistency(ptm, pauli_basis)
        vectors = pauli_basis.vectors
    else:
        ptm_dim_hilbert = int(np.sqrt(dim))
        vectors = bases.general(ptm_dim_hilbert).vectors

    tensor = np.kron(vectors.transpose((0, 2, 1)), vectors).reshape(
        (dim, dim, dim, dim), order='F')
    choi = np.einsum('ij, ijkl -> kl', ptm, tensor, optimize=True).real
    return choi


def choi_to_ptm(choi, pauli_basis=None):
    """Converts a choi matrix representation (Choi) of a process to the corresponding Pauli transfer matrix representation (PTM).

    Parameters
    ----------
    choi : ndarray
        The Choi of the process
    pauli_basis : PauliBasis, optional
        The Pauli basis used for the representations (the default is None, which corrsponds to an automatically generated general Pauli basis)
    subs_dim_hilbert : ndarray, optional
        Array of the hilbert dimensionalities of each subsystem (the default is None, which corresponds to 1 subsystem with the same dimensionality as that of the Kraus operator)

    Raises
    ------
    ValueError
        If the Choi is not a 2-dimensional square matrix.
    ValueError
        If the provided PauliBasis' dimensions do not match the dimensions of the provided Choi
    ValueError
        If the provided subspaces' hilbert dimensions do not match the dimension of the system (as set by the Choi)

    Returns
    -------
    ndarray
        The PTM of the process
    """

    _check_ptm_or_choi_dims(choi)
    dim = choi.shape[0]

    if pauli_basis is not None:
        _check_ptm_or_choi_basis_consistency(choi, pauli_basis)
        vectors = pauli_basis.vectors
    else:
        choi_dim_hilbert = int(np.sqrt(dim))
        assert dim == choi_dim_hilbert*choi_dim_hilbert
        vectors = bases.general(choi_dim_hilbert).vectors

    tensor = np.kron(vectors.transpose((0, 2, 1)), vectors).reshape(
        (dim, dim, dim, dim), order='F')

    product = np.einsum('ij, lmjk -> lmik', choi, tensor, optimize=True)
    ptm = np.einsum('lmii-> lm', product, optimize=True).real
    return ptm


def choi_to_kraus(choi):
    """Decomposes a given Choi matrix repsenentation (Choi) of a process to the correspond Kraus matrices (Kraus)

    Parameters
    ----------
    choi : ndarray
        The Choi of the process

    Raises
    ------
    ValueError
        If the Choi is not a 2-dimensional square matrix.

    Returns
    -------
    ndarray
        The array containing the decomposed Kraus matrices.
    """
    _check_ptm_or_choi_dims(choi)
    dim = choi.shape[0]

    dim_hilbert = int(np.sqrt(dim))

    einvals, einvecs = np.linalg.eig(choi)
    kraus = np.einsum("i, ijk -> ikj", np.sqrt(einvals.astype(complex)),
                      einvecs.T.reshape(dim, dim_hilbert, dim_hilbert))
    return kraus


def ptm_to_kraus(ptm):
    '''Decomposes the Pauli transfer matrix (PTM) of a process to the corrsponding Kraus operators (Kraus) by first converting the PTM to a Choi matrix representation (Choi) and then decomposing the Choi to it's eigenvectors in order to find the Kraus operators

    Parameters
    ----------
    ptm : ndarray
        The PTM of the process

    Returns
    -------
    ndarray
        The array of the Kraus operators
    '''

    choi = ptm_to_choi(ptm)
    kraus = choi_to_kraus(choi)
    return kraus


def kraus_to_choi(kraus):
    """Converts a kraus operator or an array of Kraus operators representing a process to the corresponding Choi matrix representation (Choi)

    Parameters
    ----------
    kraus : ndarray
        Either the square kraus operator or an array of square kraus operators representing a process.

    Raises
    ------
    ValueError
        If kraus is not a 2-dimensional or 3-dimensional matrix
    ValueError
        If the provided Kraus matrices are not all square

    Returns
    -------
    ndarray
        The Choi of the process
    """

    kraus = _check_kraus_dims(kraus)
    dim = kraus.shape[1]

    dim_pauli = dim * dim
    choi = np.einsum("ijk, ilm -> kjml", kraus, kraus.conj()
                     ).reshape(dim_pauli, dim_pauli)
    return choi


def convert_ptm_basis(ptm, cur_basis, new_basis):
    """Function to change the pauli basis of a Pauli transfer matrix (PTM) from the current one to a new basis

    Parameters
    ----------
    ptm : ndarray
        The PTM
    cur_basis : PauliBasis
        The current basis used for the PTM
    new_basis : PauliBasis
        The new basis the the PTM will be expanded in

    Raises
    ------
    ValueError
        If PTM is not a square matrix
    ValueError
        If the provided cur_matrix does not match the dimensions of the PTM
    ValueError
        If the new basis does not match the dimensions of the PTM

    Returns
    -------
    ndarray
        The PTM expressed in the new basis
    """

    _check_ptm_or_choi_dims(ptm)
    dim = ptm.shape[0]

    _check_ptm_or_choi_basis_consistency(ptm, cur_basis)
    cur_vectors = cur_basis.vectors

    _check_ptm_or_choi_basis_consistency(ptm, new_basis)
    new_vectors = new_basis.vectors

    converted_ptm = np.einsum("xij, yji, yz, zkl, wlk -> xw", new_vectors,
                              cur_vectors, ptm, cur_vectors, new_vectors, optimize=True).real

    return converted_ptm


def _check_kraus_dims(kraus):
    assert isinstance(kraus, np.ndarray)
    if kraus.ndim not in (2, 3):
        raise ValueError(ERR_MSGS['wrong_dim'].format(kraus.ndim))

    # If a single Kraus extend dimension by one
    if kraus.ndim == 2:
        kraus = np.array([kraus])

    dim_hilbert = kraus.shape[1]
    if kraus.shape[1:] != (dim_hilbert, dim_hilbert):
        raise ValueError(ERR_MSGS['not_sqr'].format('Kraus', kraus.shape))

    return kraus


def _check_ptm_or_choi_dims(matrix):
    assert isinstance(matrix, np.ndarray)
    dim_pauli = matrix.shape[0]
    if dim_pauli == 1:
        raise ValueError(ERR_MSGS['wrong_dim'].format(matrix.shape))

    if matrix.shape != (dim_pauli, dim_pauli):
        raise ValueError(ERR_MSGS['not_sqr'].format(matrix.shape))

    dim_hilbert = int(np.sqrt(dim_pauli))
    if dim_pauli != dim_hilbert*dim_hilbert:
        raise ValueError(ERR_MSGS['pauli_not_sqr'].format(dim_pauli))


def _check_ptm_or_choi_basis_consistency(mat, basis):
    mat_dim_pauli = mat.shape[0]
    basis_dim_pauli = basis.dim_pauli
    if mat_dim_pauli != basis_dim_pauli:
        raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
            mat_dim_pauli, basis_dim_pauli))


def _check_kraus_basis_consistency(kraus, basis):
    kraus_dim_hilbert = kraus.shape[1]
    basis_dim_hilbert = basis.dim_hilbert
    if kraus_dim_hilbert != basis_dim_hilbert:
        raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
            kraus_dim_hilbert, basis_dim_hilbert))
