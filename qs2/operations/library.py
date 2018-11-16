import numpy as np
from qs2.basis import basis
from .operation import TracePreservingOperation


# TODO: Consider a how exactly to extend this to single multi-level qubits, just np.stack?

# TODO: Consider renaming the basis sub module to something other, as basis is a short sweet word to use for parameters. Alternatively we forever give up on importing the submodule as a whole

# TODO: decide on def params for the rotate_euler

def rotate_euler(phi, theta, lamda, pauli_basis=None):
    '''Generates the operation, represented by the correspond Pauli Transfer Matrix in some basis, that perform a euler rotation on the qubit.

    Arguments:
        phi {[type]} -- [description]
        theta {[type]} -- [description]
        lamda {[type]} -- [description]

    Keyword Arguments:
        pauli_basis {PauliBasis} -- the explicitly defined basis (default: {None})

    Returns:
        Operation -- the operation
    '''

    unitary = np.array([[np.cos(theta / 2),
                         -1j * np.exp(1j * lamda) * np.sin(theta / 2)],
                        [-1j * np.exp(1j * phi) * np.sin(theta / 2),
                         np.exp(1j * (lamda + phi)) * np.cos(theta / 2)]])
    if pauli_basis is not None:
        if pauli_basis.dim_hilbert != 2:
            raise NotImplementedError
    else:
        pauli_basis = basis.general(2)

    return TracePreservingOperation(kraus=unitary, basis=pauli_basis)


def rotate_x(angle=np.pi, pauli_basis=None):
    '''Generates the operation, represented by the correspond Pauli Transfer Matrix in some basis, that perform a rotation around the x-axis by some angle.

    Keyword Arguments:
        angle {float} -- the angle of rotation (default: {np.pi})
        pauli_basis {PauliBasis} -- the explicitly defined basis (default: {None})

    Raises:
        NotImplementedError -- for now :D

    Returns:
        Operation -- the operation
    '''

    isin, cos = -1j*np.sin(angle / 2), np.cos(angle / 2)
    unitary = np.array([[cos, isin], [isin, cos]])

    if pauli_basis is not None:
        if pauli_basis.dim_hilbert != 2:
            raise NotImplementedError
    else:
        pauli_basis = basis.general(2)

    return TracePreservingOperation(kraus=unitary, basis=pauli_basis)


def rotate_y(angle=np.pi, pauli_basis=None):
    '''Generates the operation, represented by the correspond Pauli Transfer Matrix in some basis, that perform a rotation around the y-axis by some angle.

    Keyword Arguments:
        angle {float} -- the angle of rotation (default: {np.pi})
        pauli_basis {PauliBasis} -- the explicitly defined basis (default: {None})

    Raises:
        NotImplementedError -- for now :D

    Returns:
        Operation -- the operation
    '''
    sin, cos = -np.sin(angle / 2), np.cos(angle / 2)
    unitary = np.array([[cos, sin], [sin, cos]])

    if pauli_basis is not None:
        if pauli_basis.dim_hilbert != 2:
            raise NotImplementedError
    else:
        pauli_basis = basis.general(2)

    return TracePreservingOperation(kraus=unitary, basis=pauli_basis)


def rotate_z(angle=np.pi, pauli_basis=None):
    '''Generates the operation, represented by the correspond Pauli Transfer Matrix in some basis, that perform a rotation around the z-axis by some angle.

    Keyword Arguments:
        angle {float} -- the angle of rotation (default: {np.pi})
        pauli_basis {PauliBasis} -- the explicitly defined basis (default: {None})

    Raises:
        NotImplementedError -- for now :D

    Returns:
        Operation -- the operation
    '''
    exp = np.exp(-.5j * angle)
    unitary = np.array([[exp, 0], [0, exp.conj()]])

    if pauli_basis is not None:
        if pauli_basis.dim_hilbert != 2:
            raise NotImplementedError
    else:
        pauli_basis = basis.general(2)

    return TracePreservingOperation(kraus=unitary, basis=pauli_basis)
