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

         U = R_Z(\\alpha_{z1}) \\cdot R_X(\\alpha_x) \\cdot R_Z(\\alpha_{z2})

    Parameters
    ----------
    qubit: hashable
        Qubit tag

    Returns
    -------
    Gate
        An operation, that corresponds to the rotation.
    """
    def _rotate_euler(angle_z1, angle_x, angle_z2):
        exp_phi, exp_lambda = np.exp(1j * angle_z1), np.exp(1j * angle_z2)
        sin_theta, cos_theta = np.sin(angle_x / 2), np.cos(angle_x / 2)
        return kraus_to_ptm(np.array([[
            [cos_theta, -1j * exp_lambda * sin_theta],
            [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta]
        ]]), basis, basis)

    return Gate(qubit, DIM, _rotate_euler, duration=0,
                plot_metadata={"style": "box",
                               "label": "$R_({angle_z1}, {angle_x}, {angle_z2})$"},
                repr_="RotEuler({angle_z1}, {angle_x}, {angle_z2})")


def hadamard(qubit):
    """A perfect Hadamard operation.

    Returns
    -------
    Operation.from_kraus
        An operation, that corresponds to the rotation.
    """
    matrix = kraus_to_ptm(np.sqrt(0.5)*np.array([[[1, 1], [1, -1]]]), basis, basis)

    return Gate(qubit,
                DIM,
                lambda: matrix,
                duration=0,
                plot_metadata={"style": "box", "label": "$H$"},
                repr_="H")


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
    def _rotate_x(angle):
        sin, cos = np.sin(angle / 2), np.cos(angle / 2)
        return kraus_to_ptm(np.array([[[cos, -1j * sin], [-1j * sin, cos]]]),
                            basis, basis), basis, basis

    return Gate(qubit,
                DIM,
                _rotate_x,
                duration=0,
                plot_metadata={"style": "box", "label": "$X({angle})$"},
                repr_="X({angle})")


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
    def _rotate_y(angle):
        sin, cos = np.sin(angle / 2), np.cos(angle / 2)
        return kraus_to_ptm(np.array([[[cos, -sin], [sin, cos]]]), basis, basis), \
               basis, basis

    return Gate(qubit,
                DIM,
                _rotate_y,
                duration=0,
                plot_metadata={"style": "box", "label": "$Y({angle})$"},
                repr_="Y({angle})")


def rotate_z(qubit):
    """A perfect single qubit rotation around :math:`Oz` axis.

    Parameters
    ----------
    qubit: hashable
        Qubit tag

    Returns
    -------
    Gate
        An operation, that corresponds to the rotation.
    """
    def _rotate_z(angle):
        exp = np.exp(-1j * angle / 2)
        return kraus_to_ptm(np.diag([exp, exp.conj()]), basis, basis), \
           basis, basis

    return Gate(qubit,
                DIM,
                _rotate_z,
                duration=0,
                plot_metadata={"style": "box", "label": "$Z({angle})$"},
                repr_="Z({angle})")


def cphase(qubit1, qubit2):
    """A perfect controlled phase rotation.

    Parameters
    ----------
    qubit1, qubit2: hashable
        Qubit tags

    Returns
    -------
    Gate
        An operation, that corresponds to the CPhase.
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


def cnot(control_qubit, target_qubit):
    """Conditional NOT on the target qubit depending on the state of the control
    qubit..
    """
    matrix = kraus_to_ptm(np.array([[[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 1],
                                     [0, 0, 1, 0]]]), basis2, basis2)
    return Gate([control_qubit, target_qubit],
                DIM,
                lambda: (matrix, basis2, basis2),
                duration=0,
                plot_metadata={
                    "style": "line",
                    "markers": [
                        {"style": "marker", "label": "o"},
                        {"style": "marker", "label": r"$\oplus$"},
                    ],
                },
                repr_='CNot')


def swap(qubit1, qubit2):
    """A perfect SWAP gate

    Parameters
    ----------
    qubit1, qubit2: hashable
        Qubit tags

    Returns
    -------
    Gate
    """
    matrix = kraus_to_ptm(np.array([[[1, 0, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 1]]]), basis2, basis2)
    return Gate([qubit1, qubit2],
                DIM,
                lambda: (matrix, basis, basis),
                duration=0,
                plot_metadata={
                    "style": "line",
                    "markers": [
                        {"style": "marker", "label": "x"},
                        {"style": "marker", "label": "x"},
                    ],
                },
                repr_='Swap',
    )
