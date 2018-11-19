import numpy as np

from itertools import count
from functools import lru_cache
from .pauli_basis import PauliBasis

_sqrt2i = np.sqrt(0.5)


@lru_cache(maxsize=64)
def general(dim_hilbert):
    """The vector of 'Pauli matrices' in dimension n.

    First `n` matrices are matrices with 1 at the position [i, i] and zeros
    elsewhere. They are followed by a pairs of :math:`\\sigma_x`-like and
    :math:`\\sigma_y`-like matrices.

    Parameters
    ----------
    dim_hilbert: int
        Hilbert dimensionality of the space

    Returns
    -------
    PauliBasis
        Full orthonromal Pauli basis.
    """
    vectors = np.zeros((dim_hilbert * dim_hilbert, dim_hilbert, dim_hilbert),
                       dtype=complex)
    # noinspection PyTypeChecker
    labels = np.full(dim_hilbert * dim_hilbert, None, dtype=object)
    counter = count()

    for i in range(dim_hilbert):
        num = next(counter)
        vectors[num, i, i] = 1
        labels[num] = str(i)

    for i in range(dim_hilbert):
        for j in range(i):
            # x-like
            num = next(counter)
            vectors[num, i, j] = _sqrt2i
            vectors[num, j, i] = _sqrt2i
            labels[num] = "X{}{}".format(i, j)

            # y-like
            num = next(counter)
            vectors[num, i, j] = 1j * _sqrt2i
            vectors[num, j, i] = -1j * _sqrt2i
            labels[num] = "Y{}{}".format(i, j)

    return PauliBasis(vectors, labels)


@lru_cache(maxsize=64)
def gell_mann(dim_hilbert):
    """A Pauli basis consisting of the generalization of Pauli matrices for
    higher dimensions, the generalized Gell-Mann matrices.

    Gell-Mann matrices are Hermitian and traceless, except the first,
    which is the identity [1]_ [2]_.

    `qs2.basis.gell_mann(2)` is the same as `qs2.basis.twolevel_ixyz`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
    .. [2] https://en.wikipedia.org/wiki/Gell-Mann_matrices
    """

    def diagonal(index, zeros):
        if index == 0:
            diag = [np.ones(dim_hilbert) / np.sqrt(dim_hilbert)]
        else:
            diag = np.zeros(dim_hilbert)
            diag[:index] = 1
            diag[index] = -index
            diag /= np.sqrt(index * (index + 1))

        for i, d in enumerate(diag):
            zeros[i, i] = d

    def off_diagonal(i, j, zeros):
        if i < j:
            zeros[i, j] = _sqrt2i
            zeros[j, i] = _sqrt2i
        else:
            zeros[i, j] = 1j * _sqrt2i
            zeros[j, i] = -1j * _sqrt2i

    vectors = np.zeros((dim_hilbert * dim_hilbert, dim_hilbert, dim_hilbert),
                       dtype=complex)
    # noinspection PyTypeChecker
    labels = np.full(dim_hilbert * dim_hilbert, None, dtype=object)
    counter = count()

    for i in range(dim_hilbert):
        for j in range(dim_hilbert):
            num = next(counter)
            labels[num] = ("γ{}{}".format(i, j))
            if i == j:
                diagonal(i, vectors[num])
            else:
                off_diagonal(i, j, vectors[num])

    return PauliBasis(vectors, labels)


twolevel_0xy1 = PauliBasis(
    vectors=np.array([[[1, 0], [0, 0]],
                      _sqrt2i * np.array([[0, 1], [1, 0]]),
                      _sqrt2i * np.array([[0, -1j], [1j, 0]]),
                      [[0, 0], [0, 1]]]),
    labels=np.array(("0", "X", "Y", "1"), dtype=object)
)

twolevel_ixyz = PauliBasis(
    vectors=_sqrt2i * np.array([[[1, 0], [0, 1]],
                                [[0, 1], [1, 0]],
                                [[0, -1j], [1j, 0]],
                                [[1, 0], [0, -1]]]),
    labels=np.array(("I", "X", "Y", "Z"), dtype=object)
)
