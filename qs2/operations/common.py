import numpy as np
from qs2.basis import basis

ERR_MSGS = dict(
    basis_dim_mismatch='The dimensions of the given basis do not match the provided operators: operator shape is {}, while basis has dimensions {}',
    not_sqr='Only square {} matrices can be transformed: provided matrix shape is {}',
    wrong_dim='Incorred dimensionality of the provided operator: operator dimensions are: {}'
)


def kraus_to_ptm(kraus, pauli_basis=None, subs_dim_hilbert=None):
    """Converts a kraus matrix or an array of kraus matrices to the Pauli transfer matrix (PTM) representation.

    Parameters
    ----------
    kraus : ndarray
        Either the square kraus operator or an array of square kraus operators representing a process.
    pauli_basis : PauliBasis, optional
        The provided pauli basis in which the PTM is expanded (the default is None, which corrsponds to an automatically generated general Pauli basis)
    subs_dim_hilbert : ndarray, optional
        Array of the hilbert dimensionalities of each subsystem (the default is None, which corresponds to 1 subsystem with the same dimensionality as that of the Kraus operator)

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

    if kraus.ndim not in (2, 3):
        raise ValueError(ERR_MSGS['wrong_dim'].format(kraus.ndim))

    # If a single Kraus extend dimension by one
    if kraus.ndim == 2:
        kraus = np.array([kraus])

    dim = kraus.shape[1]
    if kraus.shape[1:] != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('Kraus', kraus.shape))

    # Generate the basis vectors of the full system
    if pauli_basis is not None:
        basis_dim = pauli_basis.dim_hilbert
        if dim != basis_dim:
            raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                kraus.shape, basis_dim))
        vectors = pauli_basis.vectors
    else:
        if subs_dim_hilbert:
            basis_dim = np.prod(subs_dim_hilbert)
            if dim != basis_dim:
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    kraus.shape, basis_dim))
            vectors = np.prod([basis.general(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            vectors = basis.general(dim).vectors

    ptm = np.einsum("xab, zbc, ycd, zad -> xy", vectors, kraus,
                    vectors, kraus.conj(), optimize=True).real
    return ptm


def ptm_to_choi(ptm, pauli_basis=None, subs_dim_hilbert=None):
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
    dim = ptm.shape[0]
    if ptm.shape != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('PTM', ptm.shape))

    if pauli_basis is not None:
        basis_dim = pauli_basis.dim_pauli
        if dim != basis_dim:
            raise ValueError(
                ERR_MSGS['basis_dim_mismatch'].format(ptm.shape, basis_dim))
        vectors = pauli_basis.vectors
    else:
        if subs_dim_hilbert:
            basis_dim = np.prod(subs_dim_hilbert)**2
            if dim != basis_dim:
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    ptm.shape, basis_dim))
            vectors = np.prod([basis.general(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            ptm_dim_hilbert = int(np.sqrt(dim))
            assert dim == ptm_dim_hilbert*ptm_dim_hilbert
            vectors = basis.general(ptm_dim_hilbert).vectors

    tensor = np.kron(vectors.transpose((0, 2, 1)), vectors).reshape(
        (dim, dim, dim, dim), order='F')
    choi = np.einsum('ij, ijkl -> kl', ptm, tensor, optimize=True).real
    return choi


def choi_to_ptm(choi, pauli_basis=None, subs_dim_hilbert=None):
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

    dim = choi.shape[0]
    if choi.shape != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('Choi', choi.shape))

    if pauli_basis is not None:
        basis_dim = pauli_basis.dim_pauli
        if dim != basis_dim:
            raise ValueError(
                ERR_MSGS['basis_dim_mismatch'].format(choi.shape, basis_dim))
        vectors = pauli_basis.vectors
    else:
        if subs_dim_hilbert:
            basis_dim = np.prod(subs_dim_hilbert)**2
            if dim != basis_dim:
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    choi.shape, basis_dim))
            vectors = np.prod([basis.general(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            choi_dim_hilbert = int(np.sqrt(dim))
            assert dim == choi_dim_hilbert*choi_dim_hilbert
            vectors = basis.general(choi_dim_hilbert).vectors

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
    dim = choi.shape[0]
    if choi.shape != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('Choi', choi.shape))

    dim_hilbert = int(np.sqrt(dim))
    assert dim == dim_hilbert*dim_hilbert

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

    if kraus.ndim not in (2, 3):
        raise ValueError(ERR_MSGS['wrong_dim'].format(kraus.ndim))

    # If a single Kraus extend dimension by one
    if kraus.ndim == 2:
        kraus = np.array([kraus])

    dim = kraus.shape[1]
    if kraus.shape[1:] != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('Kraus', kraus.shape))

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

    dim = ptm.shape[0]
    if ptm.shape != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('PTM', ptm.shape))

    cur_basis_dim = cur_basis.dim_pauli
    if dim != cur_basis_dim:
        raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
            ptm.shape, cur_basis_dim))
    cur_vectors = cur_basis.vectors

    new_basis_dim = new_basis.dim_pauli
    if dim != new_basis_dim:
        raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
            ptm.shape, new_basis_dim))
    new_vectors = new_basis.vectors

    converted_ptm = np.einsum("xij, yji, yz, zkl, wlk -> xw", new_vectors,
                              cur_vectors, ptm, cur_vectors, new_vectors, optimize=True).real

    return converted_ptm
