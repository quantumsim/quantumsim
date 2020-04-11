import numpy as np


def process_fidelity(operation, target_operation):
    dim = operation.dim_hilbert
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

    bases_in, bases_out = operation.bases_in, operation.bases_out

    ptm = operation.ptm(bases_in, bases_out)
    target_ptm = target_operation.ptm(bases_in, bases_out)

    # NOTE: The formula implemented here is the np.trace(target_ptm.T @ ptm)/(dim**2), as given in arXiv:1202.5344
    process_fid = np.einsum("ji, ij", target_ptm, ptm)/(dim**2)
    return process_fid


def average_gate_fidelity(operation, target_operation):
    process_fid = process_fidelity(operation, target_operation)
    dim = operation.dim_hilbert
    # NOTE: The formula implemented here is given in arXiv:1202.5344, see arXiv:quant-ph/0205035v2 as well.
    gate_fid = (dim*process_fid + 1)/(dim + 1)
    return gate_fid
