import numpy as np
from functools import lru_cache
from .operation import Transformation, join
from .. import bases

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
    Transformation
        An operation, that corresponds to the rotation.
    """
    exp_phi, exp_lambda = np.exp(1j * phi), np.exp(1j * lamda)
    sin_theta, cos_theta = np.sin(theta / 2), np.cos(theta / 2)
    matrix = np.array([
        [cos_theta, -1j * exp_lambda * sin_theta],
        [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta]])
    return Transformation.from_kraus(matrix, (2,))


@lru_cache(maxsize=32)
def rotate_x(angle=np.pi):
    """A perfect single qubit rotation around :math:`Ox` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Transformation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -1j*sin], [-1j*sin, cos]])
    return Transformation.from_kraus(matrix, (2,))


@lru_cache(maxsize=32)
def rotate_y(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oy` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Transformation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -sin], [sin, cos]])
    return Transformation.from_kraus(matrix, (2,))


@lru_cache(maxsize=32)
def rotate_z(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oz` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Transformation
        An operation, that corresponds to the rotation.
    """
    exp = np.exp(-1j * angle / 2)
    matrix = np.diag([exp, exp.conj()])
    return Transformation.from_kraus(matrix, (2,))


def phase_shift(angle=np.pi):
    matrix = np.diag([1, np.exp(1j * angle)])
    return Transformation.from_kraus(matrix, (2,))


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    Transformation
        An operation, that corresponds to the rotation.
    """
    matrix = np.sqrt(0.5)*np.array([[1, 1], [1, -1]])
    return Transformation.from_kraus(matrix, (2,))


@lru_cache(maxsize=32)
def cphase(angle=np.pi):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Transformation
        An operation, that corresponds to the rotation.
    """
    matrix = np.diag([1, 1, 1, np.exp(1j * angle)])
    return Transformation.from_kraus(matrix, (2, 2))


@lru_cache(maxsize=32)
def iswap(angle=np.pi/2):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    Transformation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle), np.cos(angle)
    matrix = np.array([[1, 0, 0, 0],
                       [0, cos, 1j*sin, 0],
                       [0, 1j*sin, cos, 0],
                       [0, 0, 0, 1]])
    return Transformation.from_kraus(matrix, (2, 2))


@lru_cache(maxsize=32)
def cnot():
    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
    return Transformation.from_kraus(matrix, (2, 2))


@lru_cache(maxsize=32)
def controlled_unitary(unitary):
    dim_hilbert = unitary.shape[0]
    if unitary.shape != (dim_hilbert, dim_hilbert):
        raise ValueError("Unitary matrix must be square")
    control_block = np.eye(2)
    off_diag_block_0 = np.zeros((2, dim_hilbert))
    off_diag_block_1 = np.zeros((dim_hilbert, 2))
    matrix = np.array([[control_block, off_diag_block_0],
                       [off_diag_block_1, unitary]])
    return Transformation.from_kraus(matrix, (2, dim_hilbert))


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


@lru_cache(maxsize=32)
def amp_damping(total_rate=None, *, exc_rate=None, damp_rate=None):
    if total_rate is not None:
        kraus = np.array([[[1, 0], [0, np.sqrt(1 - damp_rate)]],
                          [[0, np.sqrt(damp_rate)], [0, 0]]])
        return Transformation.from_kraus(kraus, (2,))
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
        bases_ = (bases.gell_mann(2),)
        return Transformation.from_ptm(ptm, bases_)


@lru_cache(maxsize=32)
def phase_damping(total_rate=None, *, x_deph_rate=None,
                  y_deph_rate=None, z_deph_rate=None):
    if total_rate is not None:
        kraus = np.array([[[1, 0], [0, np.sqrt(1 - total_rate)]],
                          [[0, 0], [0, np.sqrt(total_rate)]]])
        return Transformation.from_kraus(kraus, (2,))
    else:
        if None in (x_deph_rate, y_deph_rate, z_deph_rate):
            raise ValueError(
                "Either the total_rate or the dephasing rates along each of "
                "the three axis must be provided")
        ptm = np.diag(
            [1, 1 - x_deph_rate, 1 - y_deph_rate, 1 - z_deph_rate])
        bases_ = (bases.gell_mann(2),)
        return Transformation.from_ptm(ptm, bases_)


@lru_cache(maxsize=64)
def amp_phase_damping(damp_rate, dephase_rate):
    amp_damp = amp_damping(damp_rate)
    phase_damp = phase_damping(dephase_rate)
    return join(amp_damp, phase_damp)


@lru_cache(maxsize=16)
def bit_flipping(flip_rate):
    matrix = np.array([np.sqrt(flip_rate) * _PAULI["I"],
                       np.sqrt(1 - flip_rate) * _PAULI["X"]])
    return Transformation.from_kraus(matrix, (2,))


@lru_cache(maxsize=16)
def phase_flipping(flip_rate):
    # This is actually equivalent to the phase damping
    matrix = np.array([np.sqrt(flip_rate) * _PAULI["I"],
                       np.sqrt(1 - flip_rate) * _PAULI["Z"]])
    return Transformation.from_kraus(matrix, (2,))


@lru_cache(maxsize=16)
def bit_phase_flipping(flip_rate):
    matrix = np.array([np.sqrt(flip_rate) * _PAULI["I"],
                       np.sqrt(1 - flip_rate) * _PAULI["Y"]])
    return Transformation.from_kraus(matrix, (2,))


@lru_cache(maxsize=16)
def depolarization(depolar_rate):
    rate = depolar_rate / 2
    sqrt = np.sqrt(rate)
    matrix = np.array([np.sqrt(2 - (3 * rate)) * _PAULI["I"],
                       sqrt * _PAULI["X"],
                       sqrt * _PAULI["Y"],
                       sqrt * _PAULI["Z"]])
    return Transformation.from_kraus(matrix, (2,))
