import numpy as np
from qs2.basis import basis


def kraus_to_transfer_matrix(kraus, pauli_basis=None, double_kraus=False):
    '''Transforms a kraus operator to a Pauli Transfer Matrix (PTM)

    Arguments:
        kraus {ndarray} -- The Kraus operator

    Keyword Arguments:
        pauli_basis {PauliBasis} -- the basis of the resulting ptm (default: {None})
        double_kraus {bool} -- whether the given kraus corresponds to a single-qubit or two-qubit operator (default: {False})

    Raises:
        ValueError -- if kraus is not square
        ValueError -- if mismatch between basis and kraus dimensions

    Returns:
        ndarray -- the ptm representation of the kraus
    '''

    dim = kraus.shape[0]
    if kraus.shape() != (dim, dim):
        raise ValueError(
            'Only square kraus matrices can be transformed: input matrix shape is {}'.format(kraus.shape))

    if pauli_basis is not None:
        if dim != pauli_basis.dim_hilbert:
            raise ValueError(
                'The dimensions of the given basis do not match the kraus operators: kraus_ops shape is {}, while basis has dimensions {}'.format(kraus.shape, pauli_basis.dim_hilbert))
        tensor = pauli_basis.vectors
    else:
        tensor = basis.general(dim).vectors

    # NOTE: If we agree not to have 4-dim qubits maybe we can get rid this of this flag, that will pop up in a few places
    if double_kraus:
        tensor = np.kron(tensor, tensor)

    return np.einsum("xab, bc, ycd, ad -> xy", tensor, kraus, tensor, kraus.conj(), optimize=True).real
