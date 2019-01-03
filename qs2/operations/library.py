import numpy as np
from functools import lru_cache
from .operators import UnitaryOperator, KrausOperator
from .processes import TracePreservingProcess, join
from ..bases import gell_mann

_PAULI = dict(zip(['I', 'X', 'Y', 'Z'], gell_mann(2).vectors))


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
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    exp_phi, exp_lambda = np.exp(1j * phi), np.exp(1j * lamda)
    sin_theta, cos_theta = np.sin(theta / 2), np.cos(theta / 2)
    matrix = np.array([
        [cos_theta, -1j * exp_lambda * sin_theta],
        [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta]])
    unitary = UnitaryOperator(matrix, (2,))
    return TracePreservingProcess(unitary)


@lru_cache(maxsize=32)
def rotate_x(angle=np.pi):
    """A perfect single qubit rotation around :math:`Ox` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    sin = np.sin(angle / 2)
    cos = np.cos(angle / 2)
    matrix = np.array([[cos, -1j*sin], [-1j*sin, cos]])
    operator = UnitaryOperator(matrix, (2,))

    return TracePreservingProcess(operator)


@lru_cache(maxsize=32)
def rotate_y(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oy` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    matrix = np.array([[cos, -sin], [sin, cos]])
    operator = UnitaryOperator(matrix, (2,))

    return TracePreservingProcess(operator)


@lru_cache(maxsize=32)
def rotate_z(angle=np.pi):
    """A perfect single qubit rotation around :math:`Oz` axis.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    exp = np.exp(-1j * angle / 2)
    matrix = np.array([[exp, 0], [0, exp.conj()]])
    operator = UnitaryOperator(matrix, (2,))

    return TracePreservingProcess(operator)


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    matrix = np.sqrt(0.5)*np.array([[1, 1], [1, -1]])
    unitary = UnitaryOperator(matrix, (2,))
    return TracePreservingProcess(unitary)


@lru_cache(maxsize=32)
def cphase(angle=np.pi):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    matrix = np.diag([1, 1, 1, np.exp(1j * angle)])
    unitary = UnitaryOperator(matrix, (2, 2))

    return TracePreservingProcess(unitary)


@lru_cache(maxsize=32)
def iswap(angle=np.pi/2):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    sin, cos = np.sin(angle), np.cos(angle)
    matrix = np.array([[1, 0, 0, 0],
                       [0, cos, 1j*sin, 0],
                       [0, 1j*sin, cos, 0],
                       [0, 0, 0, 1]])
    unitary = UnitaryOperator(matrix, (2, 2))

    return TracePreservingProcess(unitary)


@lru_cache(maxsize=32)
def cnot():
    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
    unitary = UnitaryOperator(matrix, (2, 2))

    return TracePreservingProcess(unitary)


@lru_cache(maxsize=32)
def amp_damping(damp_rate):
    matrices = np.array([[[1, 0], [0, np.sqrt(1 - damp_rate)]],
                         [[0, np.sqrt(damp_rate)], [0, 0]]])
    kraus = KrausOperator(matrices, (2,))

    return TracePreservingProcess(kraus)


@lru_cache(maxsize=32)
def phase_damping(dephase_rate):
    matrices = np.array([[[1, 0], [0, np.sqrt(1 - dephase_rate)]],
                         [[0, 0], [0, np.sqrt(dephase_rate)]]])
    kraus = KrausOperator(matrices, (2,))

    return TracePreservingProcess(kraus)


@lru_cache(maxsize=64)
def amp_phase_damping(damp_rate, dephase_rate):
    amp_damp = amp_damping(damp_rate)
    phase_damp = phase_damping(dephase_rate)

    return join(amp_damp, phase_damp)


@lru_cache(maxsize=16)
def bit_flipping(flip_rate):
    matrix = np.array([np.sqrt(flip_rate) * _PAULI["I"],
                       np.sqrt(1 - flip_rate) * _PAULI["X"]])
    kraus = KrausOperator(matrix, (2,))
    return TracePreservingProcess(kraus)


@lru_cache(maxsize=16)
def phase_flipping(flip_rate):
    matrix = np.array([np.sqrt(flip_rate) * _PAULI["I"],
                       np.sqrt(1 - flip_rate) * _PAULI["Z"]])
    kraus = KrausOperator(matrix, (2,))
    return TracePreservingProcess(kraus)@lru_cache(maxsize=16)


@lru_cache(maxsize=16)
def bit_phase_flipping(flip_rate):
    matrix = np.array([np.sqrt(flip_rate) * _PAULI["I"],
                       np.sqrt(1 - flip_rate) * _PAULI["Y"]])
    kraus = KrausOperator(matrix, (2,))
    return TracePreservingProcess(kraus)


@lru_cache(maxsize=16)
def depolarizing(depolar_rate):
    matrix = np.array([np.sqrt(1-(3*depolar_rate/2)) * _PAULI["I"],
                       np.sqrt(depolar_rate / 2)*_PAULI["X"],
                       np.sqrt(depolar_rate / 2)*_PAULI["Y"],
                       np.sqrt(depolar_rate / 2)*_PAULI["Z"]])

    kraus = KrausOperator(matrix, (2,))
    return TracePreservingProcess(kraus)
