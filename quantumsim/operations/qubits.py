import numpy as np

from .. import bases
from .operation import Operation

_PAULI = dict(zip(['I', 'X', 'Y', 'Z'], bases.gell_mann(2).vectors))

bases1_default = (bases.general(2),)
bases2_default = bases1_default * 2


def rotate_euler(phi, theta, lamda):
    """A perfect single qubit rotation described by three Euler angles.

    Unitary operation, that corresponds to this rotation, is:

    .. math::

         U = R_Z(\\phi) \\cdot R_X(\\theta) \\cdot R_Z(\\lambda)

    Parameters
    ----------
    phi, theta, lamda: float
        Euler rotation angles in radians.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    exp_phi, exp_lambda = np.exp(1j * phi), np.exp(1j * lamda)
    sin_theta, cos_theta = np.sin(theta / 2), np.cos(theta / 2)
    matrix = np.array([
        [cos_theta, -1j * exp_lambda * sin_theta],
        [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta]])
    return Operation.from_kraus(matrix, bases1_default)


def rotate_x(angle=np.pi):
    """A perfect single qubit rotation around :math:`Ox` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -1j*sin], [-1j*sin, cos]])
    return Operation.from_kraus(matrix, bases1_default)


def rotate_y(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oy` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -sin], [sin, cos]])
    return Operation.from_kraus(matrix, bases1_default)


def rotate_z(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oz` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    exp = np.exp(-1j * angle / 2)
    matrix = np.diag([exp, exp.conj()])
    return Operation.from_kraus(matrix, bases1_default)


def phase_shift(angle):
    matrix = np.diag([1, np.exp(1j * angle)])
    return Operation.from_kraus(matrix, bases1_default)


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    matrix = np.sqrt(0.5)*np.array([[1, 1], [1, -1]])
    return Operation.from_kraus(matrix, bases1_default)


def cphase(angle=np.pi):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    matrix = np.diag([1, 1, 1, np.exp(1j * angle)])
    return Operation.from_kraus(matrix, bases2_default)


def iswap(angle=np.pi/2):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle), np.cos(angle)
    matrix = np.array([[1, 0, 0, 0],
                       [0, cos, 1j*sin, 0],
                       [0, 1j*sin, cos, 0],
                       [0, 0, 0, 1]])
    return Operation.from_kraus(matrix, bases2_default)


def cnot():
    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
    return Operation.from_kraus(matrix, bases2_default)


def swap():
    matrix = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])
    return Operation.from_kraus(matrix, bases2_default)


def reset():
    """A perfect reset gate, projecting the qubit to the ground state.

    Returns
    -------
    Operation.from_ptm
        An operation, that corresponds to the projection.
    """
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 1
    return Operation.from_ptm(matrix, bases1_default)


def controlled_unitary(unitary):
    dim_hilbert = unitary.shape[0]
    if unitary.shape != (dim_hilbert, dim_hilbert):
        raise ValueError("Unitary matrix must be square")
    control_block = np.eye(2)
    off_diag_block_0 = np.zeros((2, dim_hilbert))
    off_diag_block_1 = np.zeros((dim_hilbert, 2))
    matrix = np.array([[control_block, off_diag_block_0],
                       [off_diag_block_1, unitary]])
    return Operation.from_kraus(matrix, bases2_default)


def controlled_rotation(angle=np.pi, axis='z'):
    if axis == 'x':
        sin, cos = np.sin(angle / 2), np.cos(angle / 2)
        matrix = np.array([[cos, -1j*sin], [-1j*sin, cos]])
    elif axis == 'y':
        sin, cos = np.sin(angle / 2), np.cos(angle / 2)
        matrix = np.array([[cos, -sin], [sin, cos]])
    elif axis == 'z':
        exp = np.exp(-1j * angle / 2)
        matrix = np.array([[exp, 0], [0, exp.conj()]])
    else:
        raise ValueError("Please provide a valid axis, got {}".format(axis))
    return controlled_unitary(matrix)


def amp_damping(decay_prob):
    kraus = np.array([[[1, 0], [0, np.sqrt(1 - decay_prob)]],
                      [[0, np.sqrt(decay_prob)], [0, 0]]])
    return Operation.from_kraus(kraus, bases1_default)


def phase_damping(deph_prob):
    kraus = np.array([[[1, 0], [0, np.sqrt(1 - deph_prob)]],
                      [[0, 0], [0, np.sqrt(deph_prob)]]])
    return Operation.from_kraus(kraus, bases1_default)


def amp_phase_damping(decay_prob, deph_prob):
    amp_damp = amp_damping(decay_prob)
    phase_damp = phase_damping(deph_prob)
    return Operation.from_sequence(amp_damp.at(0), phase_damp.at(0))


def bit_flipping(error_prob):
    matrix = np.array([np.sqrt(1 - error_prob) * _PAULI["I"],
                       np.sqrt(error_prob) * _PAULI["X"]])
    return Operation.from_kraus(matrix, bases1_default)


def phase_flipping(error_prob):
    # This is actually equivalent to the phase damping
    matrix = np.array([np.sqrt(1 - error_prob) * _PAULI["I"],
                       np.sqrt(error_prob) * _PAULI["Z"]])
    return Operation.from_kraus(matrix, bases1_default)


def bit_phase_flipping(error_prob):
    matrix = np.array([np.sqrt(1 - error_prob) * _PAULI["I"],
                       np.sqrt(error_prob) * _PAULI["Y"]])
    return Operation.from_kraus(matrix, bases1_default)


def depolarization(error_prob):
    matrix = np.array([np.sqrt(1 - error_prob) * _PAULI["I"],
                       np.sqrt(error_prob / 3) * _PAULI["X"],
                       np.sqrt(error_prob / 3) * _PAULI["Y"],
                       np.sqrt(error_prob / 3) * _PAULI["Z"]])
    return Operation.from_kraus(matrix, bases1_default)
