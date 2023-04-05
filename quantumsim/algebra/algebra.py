import numpy as np
from functools import reduce, lru_cache
from itertools import chain

#: (dict) Pauli matrix dictionary. Keys are `'I'`, `'X'`, `'Y'`, `'Z'`.
sigma = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


@lru_cache(maxsize=128)
def bases_kron(bases):
    return reduce(np.kron, [b.vectors for b in bases])


def kraus_to_ptm(kraus, bases_in, bases_out):
    dim = bases_in[0].dim_hilbert
    nq = len(bases_in)
    if nq != len(bases_out):
        raise ValueError(
            "Input and output bases must contain the same number" " of elements"
        )
    kraus = kraus.reshape([kraus.shape[0]] + [dim] * (2 * nq))
    einsum_args = []
    for i, b in enumerate(bases_out):
        einsum_args.append(b.vectors)
        einsum_args.append([4 * nq + i, 2 * i, 2 * i + 1])
    einsum_args.append(kraus)
    einsum_args.append([6 * nq] + [2 * i + 1 for i in range(2 * nq)])
    for i, b in enumerate(bases_in):
        einsum_args.append(b.vectors)
        einsum_args.append([5 * nq + i, 2 * (i + nq) + 1, 2 * (i + nq)])
    einsum_args.append(kraus.conj())
    einsum_args.append([6 * nq] + [2 * i for i in range(2 * nq)])
    einsum_args.append([4 * nq + i for i in range(2 * nq)])
    return np.einsum(*einsum_args, optimize="greedy").real


def ptm_convert_basis(ptm, bi_old, bo_old, bi_new, bo_new):
    shape = tuple(b.dim_pauli for b in chain(bo_new, bi_new))
    d_in = np.prod([b.dim_pauli for b in bi_old])
    d_out = np.prod([b.dim_pauli for b in bo_old])
    return np.einsum(
        "xij, yji, yz, zkl, wlk -> xw",
        bases_kron(bo_new),
        bases_kron(bo_old),
        ptm.reshape((d_out, d_in)),
        bases_kron(bi_old),
        bases_kron(bi_new),
        optimize="greedy",
    ).real.reshape(shape)


def dm_to_pv(dm, bases):
    n_qubits = len(bases)
    d = bases[0].dim_hilbert
    einsum_args = [dm.reshape((d,) * (2 * n_qubits)), list(range(2 * n_qubits))]
    for i, b in enumerate(bases):
        einsum_args.append(b.vectors),
        einsum_args.append([2 * n_qubits + i, i + n_qubits, i])
    return np.einsum(*einsum_args, optimize="greedy").real


def pv_to_dm(pv, bases):
    nq = len(bases)
    dim = bases[0].dim_hilbert
    einsum_args = [pv, list(range(2 * nq, 3 * nq))]
    for i, b in enumerate(bases):
        einsum_args.append(b.vectors)
        einsum_args.append([2 * nq + i, i, nq + i])
    return np.einsum(*einsum_args, optimize="greedy").reshape((dim**nq,) * 2)


def plm_lindbladian_part(lindblad_op, bases):
    """
    Compute the Lindbladian part of a Pauli Lioville matrix for a single
    Lindblad operator.

    The resulting matrix is:

    .. math::

        \\mathcal{L}^\\text{(L)}_{ij} =
        \\hat{P}^\\text{(o)}_i \\hat{L} \\hat{P}^\\text{(i)}_j \\hat{L} -
        0.5 \\hat{P}^\\text{(o)}_i \\left\\{ \\hat{L}^\\dagger \\hat{L},
        \\hat{P}^\\text{(i)}_j \\right\\}.

    See [1]_ for more information.

    Parameters
    ----------
    lindblad_op : array
        Lindblad jump operator, in units :math:`\\hbar = 1`.
    bases : tuple of quantumsim.bases.PauliBasis
        Input and output basis for the resulting PLM.

    Returns
    -------
    array

    References
    ----------
    .. [1] https://quantumsim.gitlab.io/
    """
    n = len(bases)
    einsum_args = [
        lindblad_op,
        [6 * n] + list(range(2 * n)),
        lindblad_op.conj(),
        [6 * n] + list(range(2 * n, 4 * n)),
    ]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [4 * n + i, 2 * n + i, i]]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [5 * n + i, n + i, 3 * n + i]]
    einsum_args.append(list(range(4 * n, 6 * n)))
    out = np.einsum(*einsum_args, optimize="greedy")

    einsum_args = [
        lindblad_op,
        [6 * n] + list(range(2 * n)),
        lindblad_op.conj(),
        [6 * n] + list(range(n)) + list(range(2 * n, 3 * n)),
    ]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [4 * n + i, 3 * n + i, 2 * n + i]]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [5 * n + i, n + i, 3 * n + i]]
    einsum_args.append(list(range(4 * n, 6 * n)))
    out -= 0.5 * np.einsum(*einsum_args, optimize="greedy")

    einsum_args = [
        lindblad_op,
        [6 * n] + list(range(2 * n)),
        lindblad_op.conj(),
        [6 * n] + list(range(n)) + list(range(2 * n, 3 * n)),
    ]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [4 * n + i, n + i, 3 * n + i]]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [5 * n + i, 3 * n + i, 2 * n + i]]
    einsum_args.append(list(range(4 * n, 6 * n)))
    out -= 0.5 * np.einsum(*einsum_args, optimize="greedy")

    return out


def plm_hamiltonian_part(hamiltonian, bases):
    """
    Compute the Hamiltonian part of a Pauli Liouville matrix.

    The resulting matrix is:

    .. math::

        \\mathcal{L}^\\text{(H)}_{ij} = -i \\text{tr} \\hat{P}_i \\left[
        \\hat{H}, \\hat{P}_j\\right].

    See [1]_ for more information.

    Parameters
    ----------
    hamiltonian : array
        Hamiltonian in Lindblad equation, in units :math:`\\hbar = 1`.
    bases : tuple of quantumsim.bases.PauliBasis
        Input and output bases for the resulting PLM.

    Returns
    -------
    array

    References
    ----------
    .. [1] https://quantumsim.gitlab.io/
    """
    n = len(bases)

    einsum_args = [hamiltonian, list(range(2 * n))]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [3 * n + i, 2 * n + i, i]]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [4 * n + i, n + i, 2 * n + i]]
    einsum_args.append(list(range(3 * n, 5 * n)))
    out = np.einsum(*einsum_args, optimize="greedy")

    einsum_args = [hamiltonian, list(range(2 * n))]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [3 * n + i, n + i, 2 * n + i]]
    for i, basis in enumerate(bases):
        einsum_args += [basis.vectors, [4 * n + i, 2 * n + i, i]]
    einsum_args.append(list(range(3 * n, 5 * n)))
    out -= np.einsum(*einsum_args, optimize="greedy")

    return -1j * out
