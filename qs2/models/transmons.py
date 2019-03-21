import numpy as np
from functools import lru_cache
from scipy.linalg import expm
from qs2 import bases, Operation

_PAULI = dict(zip(['I', 'X', 'Y', 'Z'], bases.gell_mann(2).vectors))


@lru_cache(maxsize=64)
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
        [cos_theta, -1j * exp_lambda * sin_theta, 0],
        [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta, 0],
        [0, 0, 1]])
    return Operation.from_kraus(matrix, 3)


@lru_cache(maxsize=32)
def rotate_x(angle=np.pi):
    """A perfect single qubit rotation around :math:`Ox` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -1j*sin, 0], [-1j*sin, cos, 0], [0, 0, 1]])
    return Operation.from_kraus(matrix, 3)


@lru_cache(maxsize=32)
def rotate_y(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oy` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    return Operation.from_kraus(matrix, 3)


@lru_cache(maxsize=32)
def rotate_z(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oz` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    exp = np.exp(-1j * angle / 2)
    matrix = np.diag([exp, exp.conj(), 1])
    return Operation.from_kraus(matrix, 3)


def phase_shift(angle=np.pi):
    matrix = np.diag([1, np.exp(1j * angle), 1])
    return Operation.from_kraus(matrix, 3)


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    s = np.sqrt(0.5)
    matrix = np.array([[s, s, 0], [s, -s, 0], [0, 0, 1]])
    return Operation.from_kraus(matrix, 3)


@lru_cache(maxsize=32)
def cphase(angle=np.pi, leakage=0.):
    """A perfect controlled phase rotation.
    First qubit is low-frequency, second qubit is high-frequency (it leaks).

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.
    leakage: float, optional
        Leakage rate of a CPhase gate

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    dcphase = np.zeros((9, 9))
    dcphase[2, 4] = 1
    dcphase[4, 2] = 1
    angle_frac = 1 - np.arcsin(np.sqrt(leakage)) / np.pi
    unitary = expm(-1j*angle*angle_frac*dcphase)
    return Operation.from_kraus(unitary, 3)


@lru_cache(maxsize=32)
def cnot():
    dcnot = np.zeros((9, 9))
    dcnot[3, 3] = 0.5
    dcnot[4, 4] = 0.5
    dcnot[3, 4] = -0.5
    dcnot[4, 3] = -0.5
    unitary = expm(-1j*np.pi*dcnot)
    return Operation.from_kraus(unitary, 3)


@lru_cache(maxsize=64)
def amp_phase_damping(duration, t1, t2):
    t_phi = 1./(1./t2 - 0.5/t1)
    op_t1 = np.sqrt(duration / t1) * np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ])
    op_t2 = np.sqrt(2 * duration / t_phi) * np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 2]
    ])
    return Operation.from_lindblad_form([op_t1, op_t2], bases.general(3))
