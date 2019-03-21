import numpy as np
from functools import reduce, lru_cache
from itertools import chain


@lru_cache(maxsize=128)
def bases_kron(bases):
    return reduce(np.kron, [b.vectors for b in bases])


def kraus_to_ptm(kraus, bases_in, bases_out):
    dim = bases_in[0].dim_hilbert
    nq = len(bases_in)
    if nq != len(bases_out):
        raise ValueError("Input and output bases must contain the same number"
                         " of elements")
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
    return np.einsum(*einsum_args, optimize=True).real


def ptm_convert_basis(ptm, bi_old, bo_old, bi_new, bo_new):
    shape = tuple(b.dim_pauli for b in chain(bo_new, bi_new))
    d_in = np.prod([b.dim_pauli for b in bi_old])
    d_out = np.prod([b.dim_pauli for b in bo_old])
    return np.einsum("xij, yji, yz, zkl, wlk -> xw",
                     bases_kron(bo_new), bases_kron(bo_old),
                     ptm.reshape((d_out, d_in)),
                     bases_kron(bi_old), bases_kron(bi_new),
                     optimize=True).real.reshape(shape)


def dm_to_pv(dm, bases):
    n_qubits = len(bases)
    d = bases[0].dim_hilbert
    einsum_args = [dm.reshape((d,) * (2 * n_qubits)), list(range(2 * n_qubits))]
    for i, b in enumerate(bases):
        einsum_args.append(b.vectors),
        einsum_args.append([2 * n_qubits + i, i + n_qubits, i])
    return np.einsum(*einsum_args, optimize=True).real


def pv_to_dm(pv, bases):
    nq = len(bases)
    dim = bases[0].dim_hilbert
    einsum_args = [pv, list(range(2 * nq, 3 * nq))]
    for i, b in enumerate(bases):
        einsum_args.append(b.vectors)
        einsum_args.append([2 * nq + i, i, nq + i])
    return np.einsum(*einsum_args, optimize=True).reshape((dim ** nq,) * 2)


# TODO Check naming.
def lindblad_plm(lindblad_ops, basis_in, basis_out):
    if len(lindblad_ops.shape) == 2:
        op = lindblad_ops
        result = np.einsum("xab, bc, ycd, ad -> xy",
                           basis_out.vectors, op,
                           basis_in.vectors, op.conj(),
                           optimize=True)
        result -= 0.5 * np.einsum("xab, cb, cd, yda -> xy",
                                  basis_out.vectors, op.conj(),
                                  op, basis_in.vectors,
                                  optimize=True)
        result -= 0.5 * np.einsum("xab, ybc, dc, da -> xy",
                                  basis_out.vectors, basis_in.vectors,
                                  op.conj(), op, optimize=True)
        return result
    else:
        results = [lindblad_plm(lop, basis_in, basis_out)
                   for lop in lindblad_ops]
        return np.sum(results, axis=0)
