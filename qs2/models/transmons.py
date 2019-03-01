import numpy as np
from functools import lru_cache
from scipy.linalg import expm
from qs2.operations import KrausOperation, PTMOperation, Chain
from qs2 import bases

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
    return KrausOperation(matrix, 3)


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
    return KrausOperation(matrix, 3)


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
    return KrausOperation(matrix, 3)


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
    return KrausOperation(matrix, 3)


def phase_shift(angle=np.pi):
    matrix = np.diag([1, np.exp(1j * angle), 1])
    return KrausOperation(matrix, 3)


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    s = np.sqrt(0.5)
    matrix = np.array([[s, s, 0], [s, -s, 0], [0, 0, 1]])
    return KrausOperation(matrix, 3)


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
    return KrausOperation(unitary, 3)


@lru_cache(maxsize=32)
def cnot():
    dcnot = np.zeros((9, 9))
    dcnot[3, 3] = 0.5
    dcnot[4, 4] = 0.5
    dcnot[3, 4] = -0.5
    dcnot[4, 3] = -0.5
    unitary = expm(-1j*np.pi*dcnot)
    return KrausOperation(unitary, 3)


@lru_cache(maxsize=32)
def amp_damping(total_rate=None, *, exc_rate=None, damp_rate=None):
    if total_rate is not None:
        kraus = np.array([
            [[1, 0, 0],
             [0, np.sqrt(1 - total_rate), 0],
             [0, 0, 1]],
            [[0, np.sqrt(total_rate), 0],
             [0, 0, 0],
             [0, 0, 1]]])
        return KrausOperation(kraus, 3)
    else:
        if None in (exc_rate, damp_rate):
            raise ValueError(
                "Either the total_rate or both the exc_rate and damp_rate "
                "must be provided")
        comb_rate = exc_rate + damp_rate
        ptm = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt((1 - comb_rate)), 0, 0],
            [0, 0, np.sqrt((1 - comb_rate)), 0],
            [2*damp_rate - comb_rate, 0, 0, 1 - comb_rate]])
        return PTMOperation(ptm, (bases.gell_mann(3).subbasis(0, 1, 3, 4),))


@lru_cache(maxsize=32)
def phase_damping(total_rate=None, *, x_deph_rate=None,
                  y_deph_rate=None, z_deph_rate=None):
    if total_rate is not None:
        kraus = np.array([[[1, 0, 0],
                           [0, np.sqrt(1 - total_rate), 0],
                           [0, 0, 1]],
                          [[0, 0, 0],
                           [0, np.sqrt(total_rate), 0],
                           [0, 0, 1]]])
        return KrausOperation(kraus, 3)
    else:
        if None in (x_deph_rate, y_deph_rate, z_deph_rate):
            raise ValueError(
                "Either the total_rate or the dephasing rates along each of "
                "the three axis must be provided")
        ptm = np.diag(
            [1, 1 - x_deph_rate, 1 - y_deph_rate, 1 - z_deph_rate])
        return PTMOperation(ptm, (bases.gell_mann(3).subbasis([0, 1, 3, 4]),))


@lru_cache(maxsize=64)
def amp_phase_damping(damp_rate, deph_rate):
    amp_damp = amp_damping(damp_rate)
    phase_damp = phase_damping(deph_rate)
    return Chain(amp_damp.at(0), phase_damp.at(0))
