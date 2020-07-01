import numpy as np
from scipy.linalg import expm
from .. import bases
from .operation import Operation
from ..algebra.tools import verify_kraus_unitarity

bases1_default = (bases.general(3),)
bases2_default = bases1_default * 2


def dephase_leak_sub(prob):
    matrix = np.array([np.sqrt(1 - prob) * np.diag([1, 1, 1]),
                       np.sqrt(prob) * np.diag([1, 1, -1])])
    return Operation.from_kraus(matrix, bases1_default)


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
    return Operation.from_kraus(matrix, bases1_default)


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
    matrix = np.array([[cos, -1j * sin, 0], [-1j * sin, cos, 0], [0, 0, 1]])
    return Operation.from_kraus(matrix, bases1_default)


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
    return Operation.from_kraus(matrix, bases1_default)


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
    return Operation.from_kraus(matrix, bases1_default)


def phase_shift(angle):
    matrix = np.diag([1, np.exp(1j * angle), 1])
    return Operation.from_kraus(matrix, bases1_default)


def hadamard():
    """A perfect Hadamard operation.

    Returns
    -------
    Operation
        An operation, that corresponds to the rotation.
    """
    s = np.sqrt(0.5)
    matrix = np.array([[s, s, 0], [s, -s, 0], [0, 0, 1]])
    return Operation.from_kraus(matrix, bases1_default)


default_cphase_params = dict(
    leakage_rate=0.,
    leakage_phase=-np.pi/2,
    leakage_mobility_rate=0.,
    leakage_mobility_phase=0.,
    phase_22=0.,
    q0_t1=np.inf,
    q0_t2=np.inf,
    q1_t1=np.inf,
    q1_t2=np.inf,
    q1_t2_int=None,
    q0_anharmonicity=0.,
    q1_anharmonicity=0.,
    rise_time=2,
    int_time=28,
    phase_corr_time=12,
    phase_corr_error=0.,
    quasistatic_flux=0.,
    sensitivity=0.,
    phase_diff_02_12=np.pi,
    phase_diff_20_21=0.,
)


def cphase(angle=np.pi, *, model='legacy', **kwargs):
    """

    Parameters
    ----------
    angle : float
        Conditional phase of a CPhase gate, default is :math:`\\pi`.
    model : str
        Error model (currently only 'legacy' and 'NetZero' is implemented).
    **kwargs
        Parameters for the error model.

    Returns
    -------
    Operation
        Resulting CPhase operation. First qubit is static (low-frequency)
        qubit,
    """
    def p(name):
        return kwargs.get(name, default_cphase_params[name])

    for param in kwargs.keys():
        if param not in default_cphase_params.keys():
            raise ValueError('Unknown model parameter: {}'.format(param))

    int_time = p('int_time')
    leakage_rate = p('leakage_rate')
    qstatic_deviation = int_time * np.pi * \
        p('sensitivity') * (p('quasistatic_flux') ** 2)
    qstatic_interf_leakage = (0.5 - (2 * leakage_rate)) * \
                             (1 - np.cos(1.5 * qstatic_deviation))
    phase_corr_error = p('phase_corr_error')

    rot_angle = angle + (1.5 * qstatic_deviation) + (2 * phase_corr_error)

    if model.lower() == 'legacy':
        cz_op = _cphase_legacy(angle, leakage_rate)
    elif model.lower() == 'netzero':
        ideal_unitary = expm(1j * _ideal_generator(
            phase_10=phase_corr_error,
            phase_01=phase_corr_error + qstatic_deviation,
            phase_11=rot_angle,
            phase_02=rot_angle,
            phase_12=p('phase_diff_02_12') - rot_angle,
            phase_20=0,
            phase_21=p('phase_diff_20_21'),
            phase_22=p('phase_22')
        ))
        noisy_unitary = expm(1j * _exchange_generator(
            leakage=4 * leakage_rate + qstatic_interf_leakage,
            leakage_phase=p('leakage_phase'),
            leakage_mobility_rate=p('leakage_mobility_rate'),
            leakage_mobility_phase=p('leakage_mobility_phase'),
        ))
        cz_unitary = ideal_unitary @ noisy_unitary
        if not verify_kraus_unitarity(cz_unitary):
            raise RuntimeError("CPhase gate is not unitary, "
                               "verify provided parameters.")
        cz_op = Operation.from_kraus(cz_unitary, bases2_default)
    else:
        raise ValueError('Unknown CZ model: {}'.format(model))
    return cz_op


def _cphase_legacy(angle=np.pi, leakage=0.):
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
    unitary = expm(-1j * angle * angle_frac * dcphase)
    return Operation.from_kraus(unitary, bases2_default)


def _ideal_generator(phase_01,
                     phase_02,
                     phase_10,
                     phase_11,
                     phase_12,
                     phase_20,
                     phase_21,
                     phase_22):
    phases = np.array([0, phase_01, phase_02, phase_10,
                       phase_11, phase_12, phase_20, phase_21, phase_22])
    generator = np.diag(phases).astype(complex)
    return generator


def _exchange_generator(leakage, leakage_phase,
                        leakage_mobility_rate, leakage_mobility_phase):
    generator = np.zeros((9, 9), dtype=complex)

    generator[2][4] = 1j * \
        np.arcsin(np.sqrt(leakage)) * np.exp(1j * leakage_phase)
    generator[4][2] = -1j * \
        np.arcsin(np.sqrt(leakage)) * np.exp(-1j * leakage_phase)

    generator[5][7] = 1j * np.arcsin(np.sqrt(leakage_mobility_rate)) * \
        np.exp(1j * leakage_mobility_phase)
    generator[7][5] = -1j * \
        np.arcsin(np.sqrt(leakage_mobility_rate)) * \
        np.exp(-1j * leakage_mobility_phase)

    return generator


def cnot():
    dcnot = np.zeros((9, 9))
    dcnot[3, 3] = 0.5
    dcnot[4, 4] = 0.5
    dcnot[3, 4] = -0.5
    dcnot[4, 3] = -0.5
    unitary = expm(-1j * np.pi * dcnot)
    return Operation.from_kraus(unitary, bases2_default)


def amp_damping(p0_up, p1_up, p1_down, p2_down):
    """
    A gate, that excites or relaxes a qubit with a certain probability.

    Parameters
    ----------
    p0_up : float
        Probability to excite to state 1, being in the state 0
    p1_up : float
        Probability to excite to state 2, being in the state 1
    p1_down : float
        Probability to relax to state 0, being in the state 1
    p2_down : float
        Probability to relax to state 1, being in the state 2

    Returns
    -------
        quantumsim.operation._PTMOperation
    """
    ptm = np.identity(9, dtype=float)
    ptm[:3, :3] = [[1. - p0_up, p1_down, 0.],
                   [p0_up, 1. - p1_down - p1_up, p2_down],
                   [0., p1_up, 1 - p2_down]]
    basis = (bases.general(3),)
    return Operation.from_ptm(ptm, basis, basis)


def meas_butterfly(p0_up, p1_up, p1_down, p2_down):
    """
    Returns a gate, that corresponds to measurement-induced excitations.
    Each measurement should be sandwiched by two of these gates (before
    and after projection. This operation dephases the qubit immediately.

    Note: if measurement-induced leakage is reported by RB, p1_up should
    be twice larger, since RB would report average probabllity for both 0
    and 1 state.

    Parameters
    ----------
    p0_up : float
        Probability to excite to state 1, being in the state 0
    p1_up : float
        Probability to excite to state 2, being in the state 1
    p1_down : float
        Probability to relax to state 0, being in the state 1
    p2_down : float
        Probability to relax to state 1, being in the state 2

    Returns
    -------
        quantumsim.operation._PTMOperation
    """
    basis = (bases.general(3).computational_subbasis(),)
    return amp_damping(0.5*p0_up, 0.5*p1_up, 0.5*p1_down,
                       0.5*p2_down).set_bases(bases_in=basis, bases_out=basis)
