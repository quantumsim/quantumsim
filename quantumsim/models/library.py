import numpy as np
from scipy.linalg import expm

from .. import bases
from ..algebra import kraus_to_ptm
from ..circuits import Gate, ResetOperation
from . import Model, Setup

BASIS21 = (bases.general(2),)
BASIS22 = BASIS21 * 2
BASIS21_CLASSICAL = (bases.general(2).subbasis([0, 1]),)

BASIS31 = (bases.general(3),)
BASIS32 = BASIS31 * 2
BASIS31_CLASSICAL = (bases.general(3).subbasis([0, 1, 2]),)


class PerfectQubitModel(Model):
    """A model for an ideal error model, where gates are instantaneous and perfect,
    while the qubits experiences no noise.
    """
    dim = 2

    def __init__(self):
        setup = Setup(
            {
                "version": "1",
                "name": "Ideal Setup",
                "setup": [],
            }
        )
        super().__init__(setup=setup)

    def rotate_x(self, qubit, **params):
        """A perfect single qubit rotation around :math:`Ox` axis.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle` sets the rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_x(angle):
            sin, cos = np.sin(angle / 2), np.cos(angle / 2)
            return kraus_to_ptm(np.array([[[cos, -1j * sin], [-1j * sin, cos]]]),
                                BASIS21, BASIS21), BASIS21, BASIS21

        gate = Gate(qubit,
                    self.dim,
                    _rotate_x,
                    duration=0,
                    plot_metadata={"style": "box", "label": "$X({angle})$"},
                    repr_="X({angle})")
        gate.set(**params)
        return gate

    def rotate_y(self, qubit, **params):
        """A perfect single qubit rotation around :math:`Oy` axis.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle` sets the rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_y(angle):
            sin, cos = np.sin(angle / 2), np.cos(angle / 2)
            return kraus_to_ptm(np.array([[[cos, -sin], [sin, cos]]]),
                                BASIS21, BASIS21), BASIS21, BASIS21

        gate = Gate(qubit,
                    self.dim,
                    _rotate_y,
                    duration=0,
                    plot_metadata={"style": "box", "label": "$Y({angle})$"},
                    repr_="Y({angle})")
        gate.set(**params)
        return gate

    def rotate_z(self, qubit, **params):
        """A perfect single qubit rotation around :math:`Oz` axis.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle` sets the rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_z(angle):
            exp = np.exp(-1j * angle / 2)
            return kraus_to_ptm(np.diag([exp, exp.conj()]).reshape((1, 2, 2)),
                                BASIS21, BASIS21), BASIS21, BASIS21

        gate = Gate(qubit,
                    self.dim,
                    _rotate_z,
                    duration=0,
                    plot_metadata={"style": "box", "label": "$Z({angle})$"},
                    repr_="Z({angle})")
        gate.set(**params)
        return gate

    def rotate_euler(self, qubit, **params):
        """A perfect single qubit rotation described by three Euler angles.

        Unitary operation, that corresponds to this rotation, is:

        .. math::

             U = R_Z(\\alpha_{z1}) \\cdot R_X(\\alpha_x) \\cdot R_Z(\\alpha_{z2})

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle_z1`, `angle_x` and `angle_z2` set
            the rotation angles, everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_euler(angle_z1, angle_x, angle_z2):
            exp_phi, exp_lambda = np.exp(1j * angle_z1), np.exp(1j * angle_z2)
            sin_theta, cos_theta = np.sin(angle_x / 2), np.cos(angle_x / 2)
            return kraus_to_ptm(np.array([[
                [cos_theta, -1j * exp_lambda * sin_theta],
                [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta]
            ]]), BASIS21, BASIS21), BASIS21, BASIS21

        gate = Gate(qubit, self.dim, _rotate_euler, duration=0,
                    plot_metadata={"style": "box",
                                   "label": "$R_({angle_z1}, {angle_x}, {angle_z2})$"},
                    repr_="RotEuler({angle_z1}, {angle_x}, {angle_z2})")
        gate.set(**params)
        return gate

    # noinspection PyUnusedLocal
    def hadamard(self, qubit, **params):
        """A perfect Hadamard operation.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. All of them are ignored.

        Returns
        -------
        Gate
        """
        matrix = kraus_to_ptm(np.sqrt(0.5)*np.array([[[1, 1], [1, -1]]]),
                              BASIS21, BASIS21)
        return Gate(qubit,
                    self.dim,
                    lambda: (matrix, BASIS21, BASIS21),
                    duration=0,
                    plot_metadata={"style": "box", "label": "$H$"},
                    repr_="H")

    # noinspection DuplicatedCode
    def cphase(self, qubit1, qubit2, **params):
        """A perfect controlled phase rotation.

        Parameters
        ----------
        qubit1, qubit2: hashable
            Qubit tags
        **params
            Named parameters for the circuit. `angle` sets the CPhase rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _cphase(angle):
            matrix = np.diag([1, 1, 1, np.exp(1j * angle)]).reshape((1, 4, 4))
            return kraus_to_ptm(matrix, BASIS22, BASIS22), BASIS22, BASIS22

        gate = Gate([qubit1, qubit2],
                    self.dim,
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
        gate.set(**params)
        return gate

    # noinspection PyUnusedLocal
    def cnot(self, control_qubit, target_qubit, **params):
        """Conditional NOT on the target qubit depending on the state of the control
        qubit..

        Parameters
        ----------
        control_qubit, target_qubit: hashable
            Qubit tags
        **params
            Named parameters for the circuit. All of them are ignored.

        Returns
        -------
        Gate
        """
        matrix = kraus_to_ptm(np.array([[[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1],
                                         [0, 0, 1, 0]]]), BASIS22, BASIS22)
        return Gate([control_qubit, target_qubit],
                    self.dim,
                    lambda: (matrix, BASIS22, BASIS22),
                    duration=0,
                    plot_metadata={
                        "style": "line",
                        "markers": [
                            {"style": "marker", "label": "o"},
                            {"style": "marker", "label": r"$\oplus$"},
                        ],
                    },
                    repr_='CNot')

    # noinspection PyUnusedLocal
    def swap(self, control_qubit, target_qubit, **params):
        """A perfect SWAP gate

        Parameters
        ----------
        control_qubit, target_qubit: hashable
            Qubit tags
        **params
            Named parameters for the circuit. All of them are ignored.

        Returns
        -------
        Gate
        """
        matrix = kraus_to_ptm(np.array([[[1, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]]]), BASIS22, BASIS22)
        return Gate([control_qubit, target_qubit],
                    self.dim,
                    lambda: (matrix, BASIS22, BASIS22),
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

    def measure(self, qubit, **params):
        """A perfect measurement gate.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `result` sets the measurement result and
            can be `0`, `1` or a string to rename it, everything else is ignored.

        Returns
        -------
        Gate
        """

        def project(result):
            if result in (0, 1):
                basis_element = (BASIS21[0].subbasis([result]),)
                return np.array([[1.]]), basis_element, basis_element
            raise ValueError("Unknown measurement result: {}".format(result))

        gate = Gate(qubit,
                    self.dim,
                    project,
                    duration=0,
                    plot_metadata={"style": "box",
                                   "label": r"$\circ\!\!\!\!\!\!\!\nearrow$"},
                    repr_='measure')
        gate.set(**params)
        return gate

    # noinspection PyUnusedLocal
    def dephase(self, qubit, **params):
        """A perfect dephasing gate.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. All of them are ignored.

        Returns
        -------
        Gate
        """

        def _dephase():
            return np.array([[[1., 0.], [0., 1.]]]),\
                   BASIS21_CLASSICAL, BASIS21_CLASSICAL

        return Gate(qubit,
                    self.dim,
                    _dephase,
                    duration=0,
                    plot_metadata={"style": "box",
                                   "label": r"$Z$"},
                    repr_='dephase')

    # noinspection PyUnusedLocal
    def reset(self, qubit, **params):
        """A perfect reset operation.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. All of them are ignored.

        Returns
        -------
        Gate
        """
        return ResetOperation([qubit], self.dim, duration=0)


class PerfectQutritModel(Model):
    """
    A model for an ideal error model, where gates are
    instantaneous and perfect, while the qubits experiences no noise.

    This class is mostly needed to construct noisy qutrit models, since third state is
    not used, if the computation is perfect.
    """
    dim = 3

    def __init__(self):
        setup = Setup(
            {
                "version": "1",
                "name": "Ideal Setup",
                "setup": [],
            }
        )
        super().__init__(setup=setup)

    def rotate_x(self, qubit, **params):
        """A perfect single qubit rotation around :math:`Ox` axis.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle` sets the rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_x(angle):
            sin, cos = np.sin(angle / 2), np.cos(angle / 2)
            matrix = np.array([[[cos, -1j * sin, 0], [-1j * sin, cos, 0], [0, 0, 1]]])
            return kraus_to_ptm(matrix, BASIS31, BASIS31), BASIS31, BASIS31

        gate = Gate(qubit,
                    self.dim,
                    _rotate_x,
                    duration=0,
                    plot_metadata={"style": "box", "label": "$X({angle})$"},
                    repr_="X({angle})")
        gate.set(**params)
        return gate

    def rotate_y(self, qubit, **params):
        """A perfect single qubit rotation around :math:`Oy` axis.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle` sets the rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_y(angle):
            sin, cos = np.sin(angle / 2), np.cos(angle / 2)
            matrix = np.array([[[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]])
            return kraus_to_ptm(matrix, BASIS31, BASIS31), BASIS31, BASIS31

        gate = Gate(qubit,
                    self.dim,
                    _rotate_y,
                    duration=0,
                    plot_metadata={"style": "box", "label": "$Y({angle})$"},
                    repr_="Y({angle})")
        gate.set(**params)
        return gate

    def rotate_z(self, qubit, **params):
        """A perfect single qubit rotation around :math:`Oz` axis.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle` sets the rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_z(angle):
            exp = np.exp(-1j * angle / 2)
            return kraus_to_ptm(np.diag([exp, exp.conj(), 1]).reshape((1, 3, 3)),
                                BASIS31, BASIS31), BASIS31, BASIS31

        gate = Gate(qubit,
                    self.dim,
                    _rotate_z,
                    duration=0,
                    plot_metadata={"style": "box", "label": "$Z({angle})$"},
                    repr_="Z({angle})")
        gate.set(**params)
        return gate

    def rotate_euler(self, qubit, **params):
        """A perfect single qubit rotation described by three Euler angles.

        Unitary operation, that corresponds to this rotation, is:

        .. math::

             U = R_Z(\\alpha_{z1}) \\cdot R_X(\\alpha_x) \\cdot R_Z(\\alpha_{z2})

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `angle_z1`, `angle_x` and `angle_z2` set
            the rotation angles, everything else is ignored.

        Returns
        -------
        Gate
        """
        def _rotate_euler(angle_z1, angle_x, angle_z2):
            exp_phi, exp_lambda = np.exp(1j * angle_z1), np.exp(1j * angle_z2)
            sin_theta, cos_theta = np.sin(angle_x / 2), np.cos(angle_x / 2)
            return kraus_to_ptm(np.array([[
                [cos_theta, -1j * exp_lambda * sin_theta, 0],
                [-1j * exp_phi * sin_theta, exp_phi * exp_lambda * cos_theta, 0],
                [0, 0, 1]
            ]]), BASIS31, BASIS31), BASIS31, BASIS31

        gate = Gate(qubit, self.dim, _rotate_euler, duration=0,
                    plot_metadata={"style": "box",
                                   "label": "$R_({angle_z1}, {angle_x}, {angle_z2})$"},
                    repr_="RotEuler({angle_z1}, {angle_x}, {angle_z2})")
        gate.set(**params)
        return gate

    # noinspection DuplicatedCode
    def cphase(self, qubit1, qubit2, **params):
        """A perfect controlled phase rotation.

        Parameters
        ----------
        qubit1, qubit2: hashable
            Qubit tags
        **params
            Named parameters for the circuit. `angle` sets the CPhase rotation angle,
            everything else is ignored.

        Returns
        -------
        Gate
        """
        def _cphase(angle=np.pi):
            generator = np.zeros((9, 9))
            generator[2, 4] = 1
            generator[4, 2] = 1
            unitary = expm(-1j * angle * generator / np.pi)
            return kraus_to_ptm(unitary.reshape(1, 9, 9),
                                BASIS32, BASIS32), BASIS32, BASIS32

        gate = Gate([qubit1, qubit2],
                    self.dim,
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
        gate.set(**params)
        return gate

    def measure(self, qubit, **params):
        """A perfect measurement gate.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. `result` sets the measurement result and
            can be `0`, `1` or a string to rename it, everything else is ignored.

        Returns
        -------
        Gate
        """

        def project(result):
            if result in (0, 1, 2):
                basis_element = (BASIS31[0].subbasis([result]),)
                return np.array([[1.]]), basis_element, basis_element
            raise ValueError("Unknown measurement result: {}".format(result))

        gate = Gate(qubit,
                    self.dim,
                    project,
                    duration=0,
                    plot_metadata={"style": "box",
                                   "label": r"$\circ\!\!\!\!\!\!\!\nearrow$"},
                    repr_='measure')
        gate.set(**params)
        return gate

    # noinspection PyUnusedLocal
    def dephase(self, qubit, **params):
        """A perfect dephasing gate.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. All of them are ignored.

        Returns
        -------
        Gate
        """

        def _dephase():
            return np.array(np.identity(3).reshape((1, 3, 3))), \
                   BASIS31_CLASSICAL, BASIS31_CLASSICAL

        return Gate(qubit,
                    self.dim,
                    _dephase,
                    duration=0,
                    plot_metadata={"style": "box",
                                   "label": r"$Z$"},
                    repr_='dephase')

    # noinspection PyUnusedLocal
    def reset(self, qubit, **params):
        """A perfect reset operation.

        Parameters
        ----------
        qubit: hashable
            Qubit tag
        **params
            Named parameters for the circuit. All of them are ignored.

        Returns
        -------
        Gate
        """
        return ResetOperation([qubit], self.dim, duration=0)
