import numpy as np
from qs2.basis import basis

ERR_MSGS = dict(
    basis_dim_mismatch='The dimensions of the given basis do not match the provided operators: operator shape is {}, while basis has dimensions {}',
    not_sqr_='Only square {} matrices can be transformed: provided matrix shape is {}',
    wrong_dim='Incorred dimensionality of the provided operator: operator dimensions are: {}'
)


def kraus_to_ptm(kraus, pauli_basis=None, subs_dim_hilbert=None):
    if kraus.ndim not in (2, 3):
        raise ValueError(ERR_MSGS['wrong_dim'].format(kraus.ndim))

    # If a single Kraus convert to list to keep generality
    if kraus.ndim == 2:
        kraus = np.array([kraus])

    dim = kraus.shape[1]
    if kraus.shape[1:] != (dim, dim):
        raise ValueError(ERR_MSGS['not_sqr'].format('Kraus', kraus.shape))

    if pauli_basis is not None:
        basis_dim = pauli_basis.dim_hilbert
        if dim != basis_dim:
            raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                kraus.shape, basis_dim))
        vectors = pauli_basis.vectors
    else:
        if subs_dim_hilbert:
            if dim != np.prod(subs_dim_hilbert):
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    kraus.shape, basis_dim))
            vectors = np.prod([basis.gell_mann(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            vectors = basis.gell_mann(dim).vectors

    ptm = np.einsum("xab, zbc, ycd, zad -> xy", vectors, kraus,
                    vectors, kraus.conj(), optimize=True).real
    return ptm


def ptm_to_choi(ptm, pauli_basis=None, subs_dim_hilbert=1):
    if ptm.ndim != 2:
        raise ValueError(ERR_MSGS['wrong_dim'].format(ptm.ndim))

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
            if dim != np.prod(subs_dim_hilbert)**2:
                raise ValueError(ERR_MSGS['basis_dim_mismatch'].format(
                    ptm.shape, basis_dim))
            vectors = np.prod([basis.gell_mann(dim_hilbert)
                               for dim_hilbert in subs_dim_hilbert])
        else:
            dim_hilbert = np.sqrt(dim)
            assert dim == dim_hilbert*dim_hilbert
            vectors = basis.gell_mann(dim_hilbert).vectors

    tensor = np.kron(vectors.transpose((0, 2, 1)), vectors).reshape(
        (dim, dim, dim, dim), order='F')
    choi = np.einsum('ij, ijkl -> kl', ptm, tensor, optimize=True).real
    return choi


def choi_to_ptm(choi):
    raise NotImplementedError


def choi_to_kraus(choi):
    raise NotImplementedError


def kraus_to_choi(kraus):
    raise NotImplementedError


def choi_to_pm(choi):
    raise NotImplementedError


def pm_to_ptm(choi):
    raise NotImplementedError
