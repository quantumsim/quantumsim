import numpy as np
from scipy.stats import unitary_group


def random_density_matrix(dim: int, seed: int):
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


