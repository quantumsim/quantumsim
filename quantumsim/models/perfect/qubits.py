from ...algebra import kraus_to_ptm
from ...circuits import Gate
from ... import bases
import numpy as np

DIM = 2
basis = (bases.general(DIM),)
basis2 = basis*2


def rotate_euler(qubit):
    """A perfect single qubit rotation described by three Euler angles.

    Unitary operation, that corresponds to this rotation, is:

    .. math::

         U = R_Z(\\phi) \\cdot R_X(\\theta) \\cdot R_Z(\\lambda)

    Parameters
    ----------
    qubit: hashable
        Qubit tag

    Returns
    -------
    Gate
        An operation, that corresponds to the rotation.
    """
    def _rotate_euler(phi, theta, lamda):
        exp_phi, exp_lambda = np.exp(1j * phi), np.exp(1j * lamda)
        sin_theta, cos_theta = np.sin(theta / 2), np.cos(theta / 2)
        return kraus_to_ptm(np.array([[
            [cos_theta, -1j * exp_lambda * sin_theta],
            [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta]
        ]]), basis, basis)

    return Gate(qubit, DIM, _rotate_euler, duration=0,
                plot_metadata={"style": "box",
                               "label": "$R_({phi}, {theta}, {lamda})$"},
                repr_="RotEuler({phi}, {theta}, {lamda})")


def rotate_x(qubit):
    """A perfect single qubit rotation around :math:`Ox` axis.

    Parameters
    ----------
    qubit: hashable
        Qubit tag

    Returns
    -------
    Gate
        An operation, that corresponds to the rotation.
    """
    def _rotate_x(theta):
        sin, cos = np.sin(theta / 2), np.cos(theta / 2)
        return kraus_to_ptm(np.array([[[cos, -1j * sin], [-1j * sin, cos]]]),
                            basis, basis), basis, basis

    return Gate(qubit,
                DIM,
                _rotate_x,
                duration=0,
                plot_metadata={"style": "box", "label": "$X({theta})$"},
                repr_="X({theta})")


def rotate_y(qubit):
    """A perfect single qubit rotation around :math:`Oy` axis.

    Parameters
    ----------
    qubit: hashable
        Qubit tag

    Returns
    -------
    Gate
        An operation, that corresponds to the rotation.
    """
    def _rotate_y(theta):
        sin, cos = np.sin(theta / 2), np.cos(theta / 2)
        return kraus_to_ptm(np.array([[[cos, -sin], [sin, cos]]]), basis, basis), \
               basis, basis

    return Gate(qubit,
                DIM,
                _rotate_y,
                duration=0,
                plot_metadata={"style": "box", "label": "$Y({theta})$"},
                repr_="Y({theta})")


def cphase(qubit1, qubit2):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    qubit1, qubit2: hashable
        Qubit tags

    Returns
    -------
    Gate
        An operation, that corresponds to the rotation.
    """
    def _cphase(angle=np.pi):
        matrix = np.diag([1, 1, 1, np.exp(1j * angle)]).reshape((1, 4, 4))
        return kraus_to_ptm(matrix, basis2, basis2), basis2, basis2

    return Gate([qubit1, qubit2],
                DIM,
                _cphase,
                duration=0,
                plot_metadata={
                    "style": "line",
                    "markers": [
                        {"style": "marker", "label": "o"},
                        {"style": "marker", "label": "o"},
                    ],
                },
                repr_="CPhase({angle})")