import numpy as np


def process_fidelity(operation, target_operation, truncate_dimensions=False):
    dim = operation.dim_hilbert

    if truncate_dimensions:
        dim = 2
        raise NotImplementedError

    if dim != target_operation.dim_hilbert:
        raise ValueError(
            'Target operation not compatible with operation \n'
            '- expected target dimensionality: {}\n'
            '- provided dimensionality: {}'.format(
                dim, target_operation.dim_hilbert))

    if operation.num_qubits != target_operation.num_qubits:
        raise ValueError(
            'Target operation not compatible with operation \n'
            '- expected {}-qubit target operation\n'
            '- provided: {}-qubit operation'.format(
                operation.num_qubits, target_operation.num_qubits))

    bases_in, bases_out = target_operation.bases_in, target_operation.bases_out

    ptm = operation.ptm(bases_in, bases_out)
    target_ptm = target_operation.ptm(bases_in, bases_out)

    # NOTE: The formula implemented here is the np.trace(target_ptm.T @ ptm)/(dim**2),
    # as given in arXiv:1202.5344
    process_fid = np.einsum("ji, ij", target_ptm, ptm)/(dim**2)
    return process_fid


def average_fidelity(operation, target_operation, truncate_dimensions=False):
    dim = operation.dim_hilbert

    if truncate_dimensions:
        dim = 2

    process_fid = process_fidelity(
        operation, target_operation, truncate_dimensions)

    # NOTE: The formula implemented here is given in arXiv:1202.5344,
    # see arXiv:quant-ph/0205035v2 as well.
    fid = (dim*process_fid + 1)/(dim + 1)
    return fid


def average_infidelity(operation, target_operation, truncate_dimensions=False):
    fid = average_fidelity(operation, target_operation, truncate_dimensions)
    return 1 - fid


def diamond_norm(operation, target_operation):
    raise NotImplementedError
