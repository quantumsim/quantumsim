import numpy as np
from scipy.stats import unitary_group


def random_hermitian_matrix(dim: int, seed: int):
    rng = np.random.RandomState(seed)
    # noinspection PyArgumentList
    diag = rng.rand(dim)
    diag /= np.sum(diag)
    dm = np.diag(diag)
    unitary = random_unitary_matrix(dim, seed+1)
    return unitary @ dm @ unitary.conj().T


def random_unitary_matrix(dim: int, seed: int):
    rng = np.random.RandomState(seed)
    return unitary_group.rvs(dim, random_state=rng)


def verify_kraus_unitarity(kraus_ops, *, tbw_tol=1e-6):
    dim_hilbert = kraus_ops.shape[1]
    op_products = np.sum([kraus.conj().T.dot(kraus)
                          for kraus in kraus_ops], axis=0)
    return np.sum(op_products)/dim_hilbert - 1 < tbw_tol


def verify_ptm_trace_pres(ptm_op, *, rtol=1e-5, atol=1e-8):
    dim_pauli = ptm_op.shape[0]
    expected_row = np.zeros(dim_pauli)
    expected_row[0] = 1
    return np.allclose(ptm_op[0], expected_row, rtol, atol)
