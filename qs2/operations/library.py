from numpy import pi
from .processes import TracePreservingProcess


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
    raise NotImplementedError()


def rotate_x(angle=pi):
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
    raise NotImplementedError()


def rotate_y(angle=pi):
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
    raise NotImplementedError()


def rotate_z(angle=pi):
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
    raise NotImplementedError()


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    raise NotImplementedError()


def cphase(angle=pi, axis='z'):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    angle: float, optional
        Rotation angle in radians. Default is :math:`\\pi`.
    axis: 'x', 'y', or 'z'
        Rotation axis.

    Returns
    -------
    TracePreservingOperation
        An operation, that corresponds to the rotation.
    """
    raise NotImplementedError()
