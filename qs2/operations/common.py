import numpy as np
from ..basis import general


def kraus_to_transfer_matrix(kraus, double_kraus=False):
    dim = kraus.shape[0]
    if kraus.shape() != (dim, dim):
        raise ValueError(
            'Only square kraus matrices can be transformed: input matrix shape is {}'.format(kraus.shape))

    tensor = general(dim)
    if double_kraus:
        tensor = np.kron(tensor, tensor)

    return np.einsum("xab, bc, ycd, ad -> xy", tensor, kraus, tensor, kraus.conj(), optimize=True).real
