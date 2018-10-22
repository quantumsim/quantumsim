# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2016 Brian Tarasinski
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize

from . import tp
from . import ptm

import functools
import copy
import warnings

def _format_angle(angle):
    multiple_of_pi = angle / np.pi
    if np.allclose(multiple_of_pi, 1):
        return r"\pi"
    elif not np.allclose(angle, 0) and np.allclose(
            np.round(1. / multiple_of_pi, 0), 1. / multiple_of_pi):
        divisor = 1 / multiple_of_pi
        return r"%s\pi/%d" % ("" if divisor > 0 else "-", abs(divisor))
    else:
        return r"%g" % angle

class Qubit:

    def __init__(self, name, t1=np.inf, t2=np.inf):
        """A Qubit with a name and amplitude damping time t1 and phase damping
        time t2,
        
        Args:
            name (string): qubit label
            t1 (float): defined as measured in a free decay experiment
            t2 (float): defined as measured in a ramsey/hahn echo experiment

        Raises:
            AssertionError: if t2 > 2*t1 (by definition of t2, we require
                t2 <= 2*t1).
        """
        self.name = name
        assert t2 <= 2 * t1
        self.t1 = max(t1, 1e-10)
        self.t2 = max(t2, 1e-10)

    def __str__(self):
        return self.name

    def make_idling_gate(self, start_time, end_time):
        """Generates a gate that decays this qubit for a period of time
        from start_time to end_time. Currently T1 and T2 decay.
        
        Args:
            start_time (float): time when gate begins
            end_time (float): time when gate ends

        Returns:
            idling_gate (circuit.AmpPhDamp): idling gate performing
                t1 and t2 decay for required period of time.

        Raises:
            AssertionError: if end_time <= start_time. We require strictly
                that gates are given different times for ordering purposes.
        """
        assert start_time < end_time
        time = (start_time + end_time) / 2
        duration = end_time - start_time

        if np.isfinite(self.t1) or np.isfinite(self.t2):
            return AmpPhDamp(self.name, time, duration, self.t1, self.t2)
        else:
            return None


class ClassicalBit(Qubit):
    '''
    A ClassicalBit is similar to a qubit, but has no T1 or T2
    and cannot be put into a density matrix.
    '''
    def __init__(self, name):
        '''
        Args:
            name: label for qubit.
        '''
        self.name = name

    def make_idling_gate(self, start_time, end_time):
        '''Extends function of same name for qubit, replacing
        it with returning nothing.

        Args:
            start_time (float): time when gate begins
            end_time (float): time when gate ends
        '''
        pass


class VariableDecoherenceQubit(Qubit):

    def __init__(self, name, base_t1, base_t2, t1s, t2s):
        """A Qubit with a name and variable t1 and t2.

        t1 is defined as measured in a free decay experiment,
        t2 is defined as measured in a ramsey/hahn echo experiment

        Note especially that you must have t2 <= 2*t1

        Args:
            base_t1 (float): t1 when time is not inside any interval
            base_t2 (float): t2 when time is not inside any interval
            t1s (list of tuples): a list of intervals
                [(start_time, end_time, t1/t2)]
        """
        self.t1s = t1s
        self.t2s = t2s
        super().__init__(name, base_t1, base_t2)

    def make_idling_gate(self, start_time, end_time):
        """Generates a gate that decays this qubit for a period of time
        from start_time to end_time. Currently T1 and T2 decay.
        
        Args:
            start_time (float): time when gate begins
            end_time (float): time when gate ends

        Returns:
            idling_gate (circuit.AmpPhDamp): idling gate performing
                t1 and t2 decay for required period of time.

        Raises:
            AssertionError: if end_time <= start_time. We require strictly
                that gates are given different times for ordering purposes.
        """
        assert start_time < end_time
        time = (start_time + end_time) / 2
        duration = end_time - start_time

        decay_rate = 1 / self.t1
        deph_rate = 1 / self.t2

        for s, e, t1 in self.t1s:
            s = max(s, start_time)
            e = min(e, end_time)
            if (s < e):
                decay_rate += (e - s) / t1 / duration

        for s, e, t2 in self.t2s:
            s = max(s, start_time)
            e = min(e, end_time)
            if (s < e):
                deph_rate += (e - s) / t2 / duration

        return AmpPhDamp(
            self.name,
            time,
            duration,
            1 / decay_rate,
            1 / deph_rate)


class Gate:
    """
    Gates are the quantumsim primitives that act on qubits.
    A gate's primary purpose is to contain a ptm or two_ptm, and
    information about where and when the gate acts.
    """

    def __init__(self,
                 time,
                 time_start=None,
                 time_end=None,
                 conditional_bit=None):
        """
        Args:
            time (float): Time when the gate occurs. Used for ordering
                the gate. Also, most gates act as infintessimal (i.e.
                the gate acts at a single point in time but is sandwiched
                between resting gates on either side), in which case
                time_start and time_end are not set and will be set
                equal to time.
            time_start (float or None): Indicates when the gate
                interaction starts. If None, defaults to time.
            time_end (float or None): Indicates when the gate
                interaction ends. If None, defaults to time.
        """
        self.is_measurement = False
        self.time = time
        self.label = r"$G"
        self.involved_qubits = []
        self.annotation = None
        self.conditional_bit = conditional_bit
        if self.conditional_bit:
            self.involved_qubits.append(self.conditional_bit)
        if time_start is None:
            time_start = time
        if time_end is None:
            time_end = time
        self.time_start = time_start
        self.time_end = time_end

        assert time_start <= time_end

    def set_time(self, time, time_start=None, time_end=None):
        """
        Sets a new time for the gate safely (i.e. making sure
        that it has the same duration as before).
        Args:
            time (float): new time for gate.
            time_start (float or None): new start time for gate. If None,
                is adjusted to maintain dt_start = time - time_start.
            time_end (float or None): new end time for gate. If None,
                is adjusted to maintain dt_end = time_end - time.
        """
        if time_start is None:
            time_start = self.time_start - self.time + time
        if time_end is None:
            time_end = self.time_end - self.time + time
        self.time_start = time_start
        self.time_end = time_end
        self.time = time

    def increment_time(self, dt):
        """
        Increments the time on the gate safely (i.e. shifting
        the start_time and end_time).
        Args:
            dt (float): amount to increment gate by
        """
        self.time += dt
        self.time_start += dt
        self.time_end += dt

    def plot_gate(self, ax, coords):
        """
        Function to plot the gate on a matplotlib axis as part of
        a circuit plot.
        Args:
            ax (matplotlib axis): the axis to plot the gate on
            coords (dict of floats): the y-values of the qubits being
                plotted.
        """
        x = self.time
        y = coords[self.involved_qubits[-1]]
        ax.text(
            x, y, self.label,
            color='k',
            ha='center',
            va='center',
            bbox=dict(ec='k', fc='w', fill=True),
        )

        if self.conditional_bit:
            y2 = coords[self.conditional_bit]
            ax.plot((x, x), (y, y2), ".--", color='k')

    def annotate_gate(self, ax, coords):
        """
        Function to add a gate annotation if one exists to a circuit
        plot.
        Args:
            ax (matplotlib axis): the axis to plot the gate on
            coords (dict of floats): the y-values of the qubits
                being plotted.
        """
        if self.annotation:
            x = self.time
            y = coords[self.involved_qubits[0]]
            ax.annotate(self.annotation, (x, y), color='r', xytext=(
                0, -15), textcoords='offset points', ha='center')

    def involves_qubit(self, bit):
        """
        Checks if a given qubit is involved in this gate.
        Args:
            bit (string): qubit label
        Returns:
            (boolean): whether this qubit is involved in this gate.
        """
        return bit in self.involved_qubits

    def apply_to(self, sdm):
        """
        Applies this gate to a density matrix.
        To be specific, this adds the gate to the queue of
        gates to be applied to the density matrix, which
        then chooses to execute them as required.

        Args:
            sdm (sparsedm.SparseDM): the density matrix to apply the
            gate to.
        """
        if self.conditional_bit is not None:
            sdm.ensure_classical(self.conditional_bit)
            if sdm.classical[self.conditional_bit] == 1:
                f = sdm.__getattribute__(self.method_name)
                f(*self.involved_qubits[1:], **self.method_params)

        else:
            f = sdm.__getattribute__(self.method_name)
            f(*self.involved_qubits, **self.method_params)


class SinglePTMGate(Gate):

    def __init__(self, bit, time, ptm, **kwargs):
        """A gate applying a Pauli Transfer Matrix `ptm` to a single qubit
        `bit` at point `time`.
        """
        super().__init__(time, **kwargs)
        self.involved_qubits.append(bit)

        self.label = "G"
        self.ptm = ptm

    def apply_to(self, sdm):
        sdm.apply_ptm(*self.involved_qubits, ptm=self.ptm)


class Hadamard(SinglePTMGate):

    def __init__(self, bit, time, **kwargs):
        """A Hadamard gate on qubit `bit` acting at a point in time `time`
        """
        super().__init__(bit, time, ptm.hadamard_ptm(), **kwargs)
        self.label = r"$H$"


class RotateX(SinglePTMGate):

    def __init__(
            self,
            bit,
            time,
            angle,
            dephasing_angle=None,
            dephasing_axis=None,
            **kwargs):
        """ A rotation around the x-axis on the bloch sphere by `angle`.
        """

        super().__init__(bit, time, None, **kwargs)
        self.dephasing_axis = dephasing_axis
        self.dephasing_angle = dephasing_angle
        self.adjust(angle)

    def set_labels(self, angle):
        self.angle = angle
        self.label = r"$R_x({})$".format(_format_angle(angle))

    def adjust(self, angle):
        p = ptm.rotate_x_ptm(angle)
        if self.dephasing_angle:
            p = np.dot(
                p,
                ptm.dephasing_ptm(
                    0,
                    self.dephasing_angle,
                    self.dephasing_angle))
        if self.dephasing_axis:
            p = np.dot(p, ptm.dephasing_ptm(self.dephasing_axis, 0, 0))
        self.ptm = p
        self.set_labels(angle)


class RotateY(SinglePTMGate):

    def __init__(
            self,
            bit,
            time,
            angle,
            dephasing_angle=None,
            dephasing_axis=None,
            **kwargs):
        """ A rotation around the y-axis on the bloch sphere by `angle`.
        """
        super().__init__(bit, time, None, **kwargs)
        self.dephasing_axis = dephasing_axis
        self.dephasing_angle = dephasing_angle
        self.adjust(angle)

    def set_labels(self, angle):
        self.angle = angle
        self.label = r"$R_y({})$".format(_format_angle(angle))

    def adjust(self, angle):
        p = ptm.rotate_y_ptm(angle)
        if self.dephasing_angle:
            p = np.dot(
                p,
                ptm.dephasing_ptm(
                    self.dephasing_angle,
                    0,
                    self.dephasing_angle))
        if self.dephasing_axis:
            p = np.dot(p, ptm.dephasing_ptm(0, self.dephasing_axis, 0))
        self.ptm = p
        self.set_labels(angle)


class RotateXY(SinglePTMGate):
    """ A rotation by :math:`\\theta` around the axis in xOy plane, specified
    by the angle :math:`\\phi`. If :math:`\\phi = 0`, this corresponds to
    :class:`RotateX`, and if :math:`\\phi = \\pi/2` -- to :class:`RotateY`.

    In terms of Euler rotations, this rotation is expressed as:

    .. math::

       R_\\text{xy}(\\phi, \\theta) = R_\\text{E}(\\phi, \\theta, -\\phi).

    Parameters
    ----------

    bit: str
        A name of the involved qubit
    time: float
        Time of a gate in circuit.
    phi: float
        An angle, that specified the rotation axis.
    theta: float
        The rotation angle.
    dephasing_angle: float
        Dephasing amplitude, that corresponds to shrinking the Bloch
        sphere perpendicular to the rotation axis.
    dephasing_axis: float
        Dephasing amplitude, that corresponds to shrinking the Bloch
        sphere along the rotation axis.
    """

    def __init__(
            self,
            bit,
            time,
            phi,
            theta,
            dephasing_angle=None,
            dephasing_axis=None,
            **kwargs):
        super().__init__(bit, time, None, **kwargs)
        self.dephasing_axis = dephasing_axis
        self.dephasing_angle = dephasing_angle
        self.adjust(phi, theta)

    def set_labels(self, phi, theta):
        self.phi = phi
        self.theta = theta
        self.label = r"$R_{{xy}}({}, {})$".format(_format_angle(phi),
                                                  _format_angle(theta))

    def adjust(self, phi, theta):
        p = ptm.rotate_euler_ptm(phi, theta, -phi)
        if self.dephasing_angle:
            p = np.dot(
                p,
                ptm.dephasing_ptm(
                    self.dephasing_angle*np.abs(np.sin(phi)),
                    self.dephasing_angle*np.abs(np.cos(phi)),
                    self.dephasing_angle))
        if self.dephasing_axis:
            p = np.dot(p, ptm.dephasing_ptm(
                self.dephasing_axis*np.abs(np.cos(phi)),
                self.dephasing_axis*np.abs(np.sin(phi)),
                0))
        self.ptm = p
        self.set_labels(phi, theta)


class RotateZ(SinglePTMGate):

    def __init__(self, bit, time, angle, dephasing=None, **kwargs):
        """ A rotation around the z-axis on the bloch sphere by `angle`.
        """
        super().__init__(bit, time, None, **kwargs)
        self.dephasing = dephasing
        self.adjust(angle)

    def set_labels(self, angle):
        self.angle = angle
        self.label = r"$R_z({})$".format(_format_angle(angle))

    def adjust(self, angle):
        p = ptm.rotate_z_ptm(angle)
        if self.dephasing:
            p = np.dot(p, ptm.dephasing_ptm(self.dephasing, self.dephasing, 0))
        self.ptm = p

        self.set_labels(angle)


class RotateEuler(SinglePTMGate):

    def __init__(self, bit, time, phi, theta,  lamda, **kwargs):
        """ A single qubit rotation described by three Euler angles
        (theta, phi, lambda)
         U = R_Z(phi).R_X(theta).R_Z(lamda)
        """
        super().__init__(bit, time, None, **kwargs)
        self.adjust(phi, theta, lamda)

    def adjust(self, phi, theta, lamda):
        self.phi = phi
        self.theta = theta
        self.lamda = lamda
        self.ptm = ptm.rotate_euler_ptm(phi, theta, lamda)
        self.label = r"$R({}, {}, {})$".format(_format_angle(phi),
                                               _format_angle(theta),
                                               _format_angle(lamda))


class IdlingGate:
    pass


class AmpPhDamp(SinglePTMGate, IdlingGate):

    def __init__(self, bit, time, duration, t1, t2, **kwargs):
        """A amplitude-and-phase damping gate (rest gate) acting at point
        `time` for duration `duration` with amplitude damping time t1 and phase
        damping t2 (t1 as measured in free decay experiments, t2 as measured in
        ramsey or echo experiments).

        Note that the gate acts at only one point in time, but acts as if the
        damping was active for the time `duration`.

        kwargs: conditional_bit

        See also: Circuit.add_waiting_gates to add these gates automatically.
        """

        if t1 <= 0:
            raise RuntimeError("t1 must be positive")
        if t2 <= 0:
            raise RuntimeError("t2 must be positive")
        if t2 > 2 * t1:
            raise RuntimeError("t2 must not be greater than 2*t1")

        self.t1 = t1
        self.t2 = t2

        self.duration = duration

        if np.allclose(t2, 2 * t1):
            t_phi = np.inf
        else:
            t_phi = 1 / (1 / t2 - 1 / (2 * t1)) / 2

        gamma = 1 - np.exp(-duration / t1)
        lamda = 1 - np.exp(-duration / t_phi)
        super().__init__(bit, time, ptm.amp_ph_damping_ptm(gamma, lamda),
                         **kwargs)
        self.label = r"$%g\,\mathrm{ns}$" % self.duration

    def plot_gate(self, ax, coords):
        x = self.time
        y = coords[self.involved_qubits[0]]

        ax.scatter((x,), (y,), color='k', marker='x')

        ax.annotate(
            self.label, (x, y), xytext=(
                x, y + 0.3), textcoords='data', ha='center')


class DepolarizingNoise(SinglePTMGate, IdlingGate):

    def __init__(self, bit, time, duration, t1, **kwargs):
        """A depolarizing noise gate with damping rate 1/t1, acting for time
        `duration`.

        kwargs: conditional_bit

        See also: Circuit.add_waiting_gates to add these gates automatically.
        """

        self.t1 = t1

        self.duration = duration

        if 't2' in kwargs:
            del kwargs['t2']

        gamma = 1 - np.exp(-duration / t1)
        super().__init__(bit, time, ptm.dephasing_ptm(gamma, gamma, gamma),
                         **kwargs)

    def plot_gate(self, ax, coords):
        ax.scatter((self.time),
                   (coords[self.involved_qubits[-1]]), color='k', marker='o')


class BitflipNoise(SinglePTMGate, IdlingGate):

    def __init__(self, bit, time, duration, t1, **kwargs):
        """A depolarizing noise gate with damping rate 1/t1, acting for time
        `duration`.

        kwargs: conditional_bit

        See also: Circuit.add_waiting_gates to add these gates automatically.
        """

        self.t1 = t1

        self.duration = duration

        if 't2' in kwargs:
            del kwargs['t2']

        gamma = 1 - np.exp(-duration / t1)
        super().__init__(bit, time, ptm.dephasing_ptm(0, gamma, gamma),
                         **kwargs)

    def plot_gate(self, ax, coords):
        ax.scatter((self.time),
                   (coords[self.involved_qubits[-1]]), color='k', marker='o')


class ButterflyGate(SinglePTMGate, IdlingGate):

    def __init__(self, bit, time, p_exc, p_dec, **kwargs):
        super().__init__(
            bit,
            time,
            ptm.gen_amp_damping_ptm(
                gamma_up=p_exc,
                gamma_down=p_dec),
            **kwargs)

        self.label = r"$\Gamma_\uparrow / \Gamma_\downarrow$"


class TwoPTMGate(Gate):

    def __init__(self,
                 bit0, bit1,
                 two_ptm, time,
                 time_start=None,
                 time_end=None,
                 **kwargs):
        """A Two qubit gate.
        """
        super().__init__(time,
                         time_start,
                         time_end,
                         **kwargs)

        self.two_ptm = two_ptm
        self.involved_qubits.append(bit0)
        self.involved_qubits.append(bit1)

    def apply_to(self, sdm):
        sdm.apply_two_ptm(*self.involved_qubits, self.two_ptm)

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[-2]
        bit1 = self.involved_qubits[-1]
        ax.scatter((self.time), (coords[bit0]), color='r')
        ax.scatter((self.time), (coords[bit1]), color='b')

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)


class CPhase(Gate):

    def __init__(self, bit0, bit1, time, **kwargs):
        """A CPhase gate acting at time `time` between bit0 and bit1 (it is
        symmetric).

        Other arguments: conditional_bit
        """
        super().__init__(time, **kwargs)
        self.involved_qubits.append(bit0)
        self.involved_qubits.append(bit1)
        self.method_name = "cphase"
        self.method_params = {}

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[-2]
        bit1 = self.involved_qubits[-1]
        ax.scatter((self.time, self.time),
                   (coords[bit0], coords[bit1]), color='k')

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)


class CNOT(TwoPTMGate):

    def __init__(self, bit0, bit1, time, **kwargs):
        """A CNOT gate acting at time `time` between bit0 and bit1
        (bit1 is the control bit).
        """
        kraus = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        p = ptm.double_kraus_to_ptm(kraus)
        super().__init__(bit0, bit1, p, time, **kwargs)

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[-2]
        bit1 = self.involved_qubits[-1]
        ax.scatter((self.time,),
                   (coords[bit1],), color='k')
        ax.scatter((self.time,),
                   (coords[bit0],), color='k', marker=r'$\oplus$', s=200)

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)


class ISwapNoisy(TwoPTMGate):

    def __init__(self, bit0, bit1, angle, time, dephase_var=0, **kwargs):
        """
        ISwap gate, described by the two qubit operator

        1  0 0 0
        0  0 i 0
        0  i 0 0
        0  0 0 1
        """
        d = np.exp(-dephase_var/2)
        d4 = np.exp(-dephase_var/8)
        self.d = d
        self.d4 = d4
        assert d <= 1
        assert d >= 0
        kraus0 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle)*d, 1j*d*np.sin(angle), 0],
            [0, 1j*d*np.sin(angle), np.cos(angle)*d, 0],
            [0, 0, 0, 1]
        ])
        kraus1 = 1j*np.sqrt(1-d**2)/2*np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        kraus2 = 1j*np.sqrt(1-d**2)/2*np.array([
            [0, 0, 0, 0],
            [0, -1, 1, 0],
            [0, 1, -1, 0],
            [0, 0, 0, 0]
        ])

        p1 = ptm.double_kraus_to_ptm(kraus0) +\
            ptm.double_kraus_to_ptm(kraus1) +\
            ptm.double_kraus_to_ptm(kraus2)

        p0 = ptm.double_kraus_to_ptm(np.diag([1, 1, d4, d4])) +\
            ptm.double_kraus_to_ptm(np.diag([0, 0, np.sqrt(1-d4**2),
                                             np.sqrt(1-d4**2)]))

        p_iswap_noisy = p0 @ p1 @ p0

        super().__init__(bit0, bit1, p_iswap_noisy, time, **kwargs)

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[-2]
        bit1 = self.involved_qubits[-1]
        ax.scatter((self.time, self.time),
                   (coords[bit0], coords[bit1]),
                   marker="x", s=80, color='b')

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)

    def adjust(self, angle):

        kraus0 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle)*self.d, 1j*self.d*np.sin(angle), 0],
            [0, 1j*self.d*np.sin(angle), np.cos(angle)*self.d, 0],
            [0, 0, 0, 1]
        ])
        kraus1 = 1j*np.sqrt(1-self.d**2)/2*np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        kraus2 = 1j*np.sqrt(1-self.d**2)/2*np.array([
            [0, 0, 0, 0],
            [0, -1, 1, 0],
            [0, 1, -1, 0],
            [0, 0, 0, 0]
        ])

        p1 = ptm.double_kraus_to_ptm(kraus0) +\
            ptm.double_kraus_to_ptm(kraus1) +\
            ptm.double_kraus_to_ptm(kraus2)

        p0 = ptm.double_kraus_to_ptm(np.diag([1, 1, self.d4, self.d4])) +\
            ptm.double_kraus_to_ptm(np.diag([0, 0, np.sqrt(1-self.d4**2),
                                             np.sqrt(1-self.d4**2)]))

        self.two_ptm = p0 @ p1 @ p0


class ISwapCoherent(TwoPTMGate):
    '''
    Class for a coherent ISwap gate. Can be operated
    by fixing the pulse amplitude and the duration
    (mode='experiment'),
    by fixing the duration and angle and solving for
    the amplitude (mode='amplitude'),
    or by fixing the amplitude and angle and solving
    for the time (mode='time').
    '''
    def __init__(self, bit0, bit1, time,
                 gap, E01, E10=None,
                 duration=None, angle=None,
                 mode='time'):
        '''
        Args:
            bit0: low-frequency qubit label
            bit1: high-frequency qubit label
            E01: energy of the low-frequency qubit (in the absence of coupling)
            E10: energy of the high-frequency qubit (in the absence of coupling)
                Fixed if mode='amplitude', and set to E01 otherwise if =None.
            duration: gate length. Fixed if mode='time', otherwise required.
                required.
            angle: rotation angle. Fixed if mode='experiment', otherwise required.
            mode: 'time', 'amplitude', 'angle'; chooses which gate parameter
                to be left free.

        '''

        self.E01 = E01
        self.E10 = E10
        self.duration = duration
        self.mode = mode
        self.angle = angle
        self.gap = gap

        unitary = self.make_unitary()

        time_start = time - self.duration/2
        time_end = time + self.duration/2

        super().__init__(bit0, bit1, ptm.double_kraus_to_ptm(unitary),
                         time, time_start=time_start, time_end=time_end)


    def _calc_angle(self, E10=None):
        '''
        Calculates the angle of rotation between the |01> and |10>
        states for the ISwap gate.
        Takes parameters stored in the class (with the possible
        exception of E10).

        Args:
            E10 (float or None, optional): The energy of the |10>
                state (with the excitation on the high-frequency qubit).
                To be set during the function for the sake of
                minimization.
        '''
        angle = self.angle
        gap = self.gap
        E01 = self.E01
        duration = self.duration
        if E10 is None:
            E10 = self.E10

        E0 = 0.5*(E01+E10)
        Delta = 0.5*(E01-E10)
        delta_E = np.sqrt(gap**2 + Delta**2)
        K_plus = delta_E - Delta
        K_minus = -delta_E - Delta
        M_plus = np.sqrt(gap**2 + K_plus**2)
        M_minus = np.sqrt(gap**2 + K_minus**2)

        # Sanity check
        assert np.isclose(gap / M_minus, K_plus / M_plus)

        ct = np.sqrt(gap**4 + K_plus**4 + 2 * gap**2 * K_plus**2 *
                     np.cos(2*delta_E*duration)) / M_plus**2
        st = gap * K_plus * 2 * np.sin(delta_E*duration) / M_plus**2

        assert np.isclose(ct**2 + st**2, 1)
        angle = np.angle(ct + 1j*st)
        return angle

    def make_unitary(self):
        '''
        Makes the unitary for an iSwap gate.
        '''

        mode = self.mode
        angle = self.angle
        gap = self.gap
        E01 = self.E01
        E10 = self.E10
        duration = self.duration

        if mode == 'experiment':
            # This mode assumes that the experimental parameters
            # are given, and that the final angle is to be found
            # out.

            E0 = 0.5*(E01+E10)
            Delta = 0.5*(E01-E10)
            delta_E = np.sqrt(gap**2 + Delta**2)
            K_plus = delta_E - Delta
            M_plus = np.sqrt(gap**2 + K_plus**2)
            angle = self._calc_angle()
            self.angle = angle

        elif mode == 'time':
            # This mode assumes that the duration of the gate
            # has not been set, and will be fixed to make
            # the angle as desired.
            if E10 is None:
                warnings.warn('E10 not set, defaulting to E10=E01')
                E10 = E01
            delta_E = np.sqrt(gap**2 + 0.25*(E01-E10)**2)
            K_plus = delta_E - 0.5*(E01-E10)
            M_plus = np.sqrt(gap**2 + K_plus**2)
            E0 = 0.5*(E01+E10)
            duration = np.arccos(
                (M_plus**4*np.cos(angle)**2 -
                 gap**4 - K_plus**4) /
                (2 * gap**2 * K_plus**2)) / (2*delta_E)
            if not np.isfinite(duration):
                raise ValueError('''
                    Cannot perform an ISwap of angle {} with detuning {}.
                    '''.format(angle, (E01-E10)/gap))
            self.duration = duration

        elif mode == 'amplitude':
            # This mode assumes that the detuning of the gate
            # has not been set, and will be fixed to make the
            # desired angle.
            if duration is None:
                raise ValueError('''Cannot use the amplitude knob
                                 without a set time''')
            # I don't think I can solve these equations analytically
            # for E10, defaulting to a numerical solution

            res = minimize(lambda x: np.abs(self.angle-self._calc_angle(x)),
                           [E01-0.1])
            E10 = res['x']
            self.E10 = E10
            delta_E = np.sqrt(gap**2 + 0.25*(E01-E10)**2)
            K_plus = delta_E - 0.5*(E01-E10)
            M_plus = np.sqrt(gap**2 + K_plus**2)
            E0 = 0.5*(E01+E10)

        unitary = np.zeros([4, 4], dtype=complex)
        unitary[0, 0] = 1
        unitary[2, 2] = np.exp(1j*E0*duration) / M_plus**2 * (
            gap**2*np.exp(1j*delta_E*duration) +
            K_plus**2*np.exp(-1j*delta_E*duration))
        unitary[1, 1] = np.exp(1j*E0*duration) / M_plus**2 * (
            gap**2*np.exp(-1j*delta_E*duration) +
            K_plus**2*np.exp(1j*delta_E*duration))
        unitary[3, 3] = np.exp(2j*E0*duration)
        unitary[1, 2] = np.exp(1j*E0*duration) * gap * K_plus / M_plus**2 * (
            np.exp(1j*delta_E*duration) - np.exp(-1j*delta_E*duration))
        unitary[2, 1] = np.exp(1j*E0*duration) * gap * K_plus / M_plus**2 * (
            np.exp(1j*delta_E*duration) - np.exp(-1j*delta_E*duration))

        return unitary

    def adjust(self, angle=None, E10=None, duration=None):
        '''
        Updates angle, E10 and/or duration of gate.
        Args:
            angle: angle of rotation (unable to be set when
                mode='experiment')
            E10: high-frequency qubit energy (unable to be
                set when mode='amplitude')
            duration: gate duration (unable to be set when
                mode='time').
        '''
        if self.mode == 'experiment':
            if E10:
                self.E10 = E10
            if duration:
                self.duration = duration
            if angle:
                warnings.warn('''You cannot set the angle
                    when mode=experiment - fix the duration
                    and detuning (E10) instead.''')
        if self.mode == 'time':

            warnings.warn('''This changes the duration of the
                gate, which will cause inconsistencies with
                idling gates. Proceed at your own caution.
                (Use mode=experiment or mode=amplitude to 
                avoid this).''')
            if E10:
                self.E10 = E10
            if angle:
                self.angle = angle
            if duration:
                warnings.warn('''You cannot set the gate duration
                    when mode=time - fix the detuning (E10) and
                    angle instead.''')

        if self.mode == 'amplitude':
            if E10:
                warnings.warn('''You cannot set the detuning
                    when mode=amplitude - fix the time and
                    angle instead.''')
            if angle:
                self.angle = angle
            if duration:
                self.duration = duration

        unitary = self.make_unitary()

        self.two_ptm = ptm.double_kraus_to_ptm(unitary)

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[-2]
        bit1 = self.involved_qubits[-1]
        ax.scatter((self.time, self.time),
                   (coords[bit0], coords[bit1]),
                   marker="x", s=80, color='b')

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)


class ISwapIncoherent(ISwapCoherent):
    '''
    Class to make an incoherent version of a coherent ISwap gate,
    by numerically integrating over the PTM.
    '''
    def __init__(self, width, num_points=19, xmin=3, **kwargs):
        '''
        @width - width of the Gaussian distribution to draw
            E10 from
        @num_points - number of points to sum over
        @xmin - distance to extend summation
        '''
        super().__init__(**kwargs)
        self.width = width
        self.num_points = num_points
        self.xmin = xmin
        self.make_ptm()

    def make_ptm(self):
        '''
        Numerically integrates a PTM to obtain an incoherent version.
        '''
        xmin = self.xmin
        num_points = self.num_points
        width = self.width

        mean_E10 = self.E10

        temp_angle = self.angle
        temp_mode = self.mode

        self.mode = 'experiment'
        self.two_ptm = np.zeros([16,16])
        for E10 in np.linspace(
                mean_E10 - xmin*width * (num_points - 1) / num_points,
                mean_E10 + xmin*width * (num_points - 1) / num_points):
            self.E10 = E10
            unitary = self.make_unitary()
            p = np.exp(-(E10-mean_E10)**2 / (2*width**2)) /\
                np.sqrt(2 * np.pi * width**2)
            self.two_ptm += ptm.double_kraus_to_ptm(unitary) * p

        self.two_ptm *= 1/self.two_ptm[0,0]

        self.mode = temp_mode
        self.angle = temp_angle
        self.E10 = mean_E10


class ISwapRotation(TwoPTMGate):

    def __init__(self, bit0, bit1, angle, time,
                 t1_bit0=None, t1_bit1=None,
                 t2_bit1=None, interaction_time=0,
                 t2_bit0_dec=None,
                 **kwargs):
        """
        ISwap rotation gate, described by the two qubit operator

        1  0                0               0
        0  cos(theta)       i*sin(theta)    0
        0  i*sin(theta)     cos(theta)      0
        0  0                0               1
        """
        '''
        time: Total gate time given by the hardware (20 ns)
        angle: Determines the swap exchange
        interaction time: Realistic iSwap interaction time, which is different
        than the total gate time.
        t2 enhanced: T2 of the system during the iSwap interaction time
        '''

        self.angle = angle
        if interaction_time == 0:

            d_var = 0
            c = np.cos(angle)
            cc = 0.5 * (1 + np.cos(2*angle))
            s = np.sin(angle)
            ss = 0.5 * (1 - np.cos(2*angle))
            sc = np.sin(angle) * np.cos(angle)

            t_start = None
            t_end = None

        else:
            if (not t2_bit0_dec) or (not t2_bit1)\
                    or (not t1_bit0) or (not t1_bit1):
                raise ValueError("""I need non-zero t1 and t2
                    for both qubits if interaction_time>0""")

            # d_var is the width of the gaussia squared
            if t2_bit0_dec == np.inf:
                """
                Ensures that for non-zero interaction time without two-qubit
                phase error we get the same result as per interaction_time=0.
                In general this will not occur, but it is useful for debugging
                purposes
                """
                d_var = 0
                t2_bit0_dec = t2_bit1
            else:
                d_var = 1 - np.exp(-interaction_time/t2_bit0_dec)

            c = np.cos(angle) * np.exp(-d_var/2)
            cc = 0.5 * (1 + np.exp(-2*d_var) * np.cos(2*angle))
            s = np.sin(angle) * np.exp(-d_var/2)
            ss = 0.5 * (1 - np.exp(-2*d_var) * np.cos(2*angle))
            sc = np.exp(-2*d_var) * np.sin(angle) * np.cos(angle)

            t_start = time - interaction_time/2
            t_end = time + interaction_time/2

            if (t1_bit0) and (t1_bit1) <= 0:
                raise RuntimeError("t1 must be positive")
            if (t2_bit0_dec) and (t2_bit1) <= 0:
                raise RuntimeError("t2 must be positive")
            if t2_bit1 > 2 * t1_bit1:
                raise RuntimeError("t2 must not be greater than 2*t1")
            if t2_bit0_dec > 2 * t1_bit0:
                raise RuntimeError("t2 must not be greater than 2*t1")

            if np.allclose(t2_bit0_dec, 2 * t1_bit0) or\
               np.allclose(t2_bit1, 2 * t1_bit1):
                t_phi_bit0 = np.inf
                t_phi_bit1 = np.inf
            else:
                t_phi_bit0 = 1 / (1 / t2_bit0_dec - 1 / (2 * t1_bit0)) / 2
                t_phi_bit1 = 1 / (1 / t2_bit1 - 1 / (2 * t1_bit1)) / 2

            gamma_bit0 = 1 - np.exp(- (interaction_time/2) / t1_bit0)
            lamda_bit0 = 1 - np.exp(- (interaction_time/2) / t_phi_bit0)

            gamma_bit1 = 1 - np.exp(- (interaction_time/2) / t1_bit1)
            lamda_bit1 = 1 - np.exp(- (interaction_time/2) / t_phi_bit1)

            PTM_bit0 = np.kron(ptm.amp_ph_damping_ptm(gamma_bit0, lamda_bit0),
                               np.identity(4))
            PTM_bit1 = np.kron(np.identity(4),
                               ptm.amp_ph_damping_ptm(gamma_bit1, lamda_bit1))

        assert d_var >= 0
        assert d_var <= 1

        p_iswap = np.array([
            [1, 0,  0,   0, 0, 0,   0,  0,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, c,  0,   0, 0, 0,   0,  0,  0,   0, 0, -s,   0, 0,  0, 0],
            [0, 0,  c,   0, 0, 0,   0,  s,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, 0,  0,  cc, 0, 0, -sc,  0,  0,  sc, 0,  0,  ss, 0,  0, 0],
            [0, 0,  0,   0, c, 0,   0,  0,  0,   0, 0,  0,   0, 0, -s, 0],
            [0, 0,  0,   0, 0, 1,   0,  0,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, 0,  0,  sc, 0, 0,  cc,  0,  0,  ss, 0,  0, -sc, 0,  0, 0],
            [0, 0, -s,   0, 0, 0,   0,  c,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, 0,  0,   0, 0, 0,   0,  0,  c,   0, 0,  0,   0, s,  0, 0],
            [0, 0,  0, -sc, 0, 0,  ss,  0,  0,  cc, 0,  0,  sc, 0,  0, 0],
            [0, 0,  0,   0, 0, 0,   0,  0,  0,   0, 1,  0,   0, 0,  0, 0],
            [0, s,  0,   0, 0, 0,   0,  0,  0,   0, 0,  c,   0, 0,  0, 0],
            [0, 0,  0,  ss, 0, 0,  sc,  0,  0, -sc, 0,  0,  cc, 0,  0, 0],
            [0, 0,  0,   0, 0, 0,   0,  0, -s,   0, 0,  0,   0, c,  0, 0],
            [0, 0,  0,   0, s, 0,   0,  0,  0,   0, 0,  0,   0, 0,  c, 0],
            [0, 0,  0,   0, 0, 0,   0,  0,  0,   0, 0,  0,   0, 0,  0, 1]
        ])

        self.d_var = d_var
        self.interaction_time = interaction_time
        self.time = time
        self.time_start = t_start
        self.time_end = t_end
        self.t1_bit0 = t1_bit0
        self.t2_bit0_dec = t2_bit0_dec
        self.t1_bit1 = t1_bit1
        self.t2_bit1 = t2_bit1

        if interaction_time == 0:
            super().__init__(bit0, bit1,
                             ptm.to_0xy1_basis(p_iswap),
                             time,
                             time_start=t_start,
                             time_end=t_end,
                             **kwargs)
        else:
            full_iswap = PTM_bit0 @ PTM_bit1 @ ptm.to_0xy1_basis(p_iswap) @\
                PTM_bit1 @ PTM_bit0

            super().__init__(bit0, bit1,
                             full_iswap,
                             time,
                             time_start=t_start,
                             time_end=t_end,
                             **kwargs)

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[-2]
        bit1 = self.involved_qubits[-1]
        ax.scatter((self.time, self.time),
                   (coords[bit0], coords[bit1]),
                   marker="x", s=80, color='b')

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)

    def adjust(self, angle):

        self.angle = angle

        if self.interaction_time == 0:
            c = np.cos(angle)
            cc = 0.5 * (1 + np.cos(2*angle))
            s = np.sin(angle)
            ss = 0.5 * (1 - np.cos(2*angle))
            sc = np.sin(angle) * np.cos(angle)
        else:
            c = np.cos(angle) * np.exp(-self.d_var/2)
            cc = 0.5 * (1 + np.exp(-2 * self.d_var) * np.cos(2*angle))
            s = np.sin(angle) * np.exp(-self.d_var/2)
            ss = 0.5 * (1 - np.exp(-2 * self.d_var) * np.cos(2*angle))
            sc = np.exp(-2*self.d_var) * np.sin(angle) * np.cos(angle)

            if (self.t1_bit0) and (self.t1_bit1) <= 0:
                raise RuntimeError("t1 must be positive")
            if (self.t2_bit0_dec) and (self.t2_bit1) <= 0:
                raise RuntimeError("t2 must be positive")
            if self.t2_bit1 > 2 * self.t1_bit1:
                raise RuntimeError("t2 must not be greater than 2*t1")
            if self.t2_bit0_dec > 2 * self.t1_bit0:
                raise RuntimeError("t2 must not be greater than 2*t1")

            if np.allclose(self.t2_bit0_dec, 2 * self.t1_bit0) or\
               np.allclose(self.t2_bit1, 2 * self.t1_bit1):
                t_phi_bit0 = np.inf
                t_phi_bit1 = np.inf
            else:
                t_phi_bit0 = 1 / (1 / self.t2_bit0_dec -
                                  1 / (2 * self.t1_bit0)) / 2
                t_phi_bit1 = 1 / (1 / self.t2_bit1 - 1 /
                                  (2 * self.t1_bit1)) / 2

            gamma_bit0 = 1 - np.exp(- (self.interaction_time/2) / self.t1_bit0)
            lamda_bit0 = 1 - np.exp(- (self.interaction_time/2) / t_phi_bit0)

            gamma_bit1 = 1 - np.exp(- (self.interaction_time/2) / self.t1_bit1)
            lamda_bit1 = 1 - np.exp(- (self.interaction_time/2) / t_phi_bit1)

            PTM_bit0 = np.kron(ptm.amp_ph_damping_ptm(gamma_bit0, lamda_bit0),
                               np.identity(4))
            PTM_bit1 = np.kron(np.identity(4),
                               ptm.amp_ph_damping_ptm(gamma_bit1, lamda_bit1))

        assert self.d_var >= 0
        assert self.d_var <= 1

        p_iswap = np.array([
            [1, 0,  0,   0, 0, 0,   0,  0,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, c,  0,   0, 0, 0,   0,  0,  0,   0, 0, -s,   0, 0,  0, 0],
            [0, 0,  c,   0, 0, 0,   0,  s,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, 0,  0,  cc, 0, 0, -sc,  0,  0,  sc, 0,  0,  ss, 0,  0, 0],
            [0, 0,  0,   0, c, 0,   0,  0,  0,   0, 0,  0,   0, 0, -s, 0],
            [0, 0,  0,   0, 0, 1,   0,  0,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, 0,  0,  sc, 0, 0,  cc,  0,  0,  ss, 0,  0, -sc, 0,  0, 0],
            [0, 0, -s,   0, 0, 0,   0,  c,  0,   0, 0,  0,   0, 0,  0, 0],
            [0, 0,  0,   0, 0, 0,   0,  0,  c,   0, 0,  0,   0, s,  0, 0],
            [0, 0,  0, -sc, 0, 0,  ss,  0,  0,  cc, 0,  0,  sc, 0,  0, 0],
            [0, 0,  0,   0, 0, 0,   0,  0,  0,   0, 1,  0,   0, 0,  0, 0],
            [0, s,  0,   0, 0, 0,   0,  0,  0,   0, 0,  c,   0, 0,  0, 0],
            [0, 0,  0,  ss, 0, 0,  sc,  0,  0, -sc, 0,  0,  cc, 0,  0, 0],
            [0, 0,  0,   0, 0, 0,   0,  0, -s,   0, 0,  0,   0, c,  0, 0],
            [0, 0,  0,   0, s, 0,   0,  0,  0,   0, 0,  0,   0, 0,  c, 0],
            [0, 0,  0,   0, 0, 0,   0,  0,  0,   0, 0,  0,   0, 0,  0, 1]
        ])

        if self.interaction_time == 0:
            self.two_ptm = ptm.to_0xy1_basis(p_iswap)
        else:
            self.two_ptm = PTM_bit0 @ PTM_bit1 @ ptm.to_0xy1_basis(p_iswap) @\
                           PTM_bit1 @ PTM_bit0


class Swap(TwoPTMGate):

    def __init__(self, bit0, bit1, time, **kwargs):
        """
        Swap gate, described by the two qubit operator

        1 0 0 0
        0 0 1 0
        0 1 0 0
        0 0 0 1
        """
        kraus = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        p = ptm.double_kraus_to_ptm(kraus)
        super().__init__(bit0, bit1, p, time, **kwargs)

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[-2]
        bit1 = self.involved_qubits[-1]
        ax.scatter((self.time, self.time),
                   (coords[bit0], coords[bit1]),
                   marker="x", s=80, color='k')

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)


class NoisyCPhase(TwoPTMGate):

    def __init__(self, bit0, bit1, time, dephase_var=0, **kwargs):

        d = np.exp(-dephase_var / 2)
        d2 = np.exp(-dephase_var / 4)
        assert d >= 0
        assert d <= 1

        p0 = ptm.double_kraus_to_ptm(np.diag([1, 1, 1, -d])) +\
            ptm.double_kraus_to_ptm(np.diag([0, 0, 0, -np.sqrt(1-d**2)]))
        p1 = ptm.double_kraus_to_ptm(np.diag([1, 1, d2, d2])) +\
            ptm.double_kraus_to_ptm(np.diag([0, 0, np.sqrt(1-d2**2),
                                             np.sqrt(1-d2**2)]))

        super().__init__(bit0, bit1, p0 @ p1, time, **kwargs)


class CPhaseRotation(TwoPTMGate):

    def __init__(self, bit0, bit1, angle, time, dephase_var=0, **kwargs):
        super().__init__(bit0, bit1, None, time, **kwargs)
        self.dephase_var = dephase_var
        self.adjust(angle)

    def adjust(self, angle):

        if angle != 0:
            d = np.exp(-self.dephase_var * (angle/np.pi)**2 / 2)
            d2 = np.exp(-self.dephase_var * (angle/np.pi)**2 / 4)
        else:
            d = 1
            d2 = 1
        assert d >= 0
        assert d <= 1

        p0 = ptm.double_kraus_to_ptm(np.diag([1, 1, 1,
                                              np.exp(1j * angle)*d])) +\
            ptm.double_kraus_to_ptm(np.diag([0, 0, 0,
                                             np.exp(1j * angle) *
                                             np.sqrt(1-d**2)]))

        p1 = ptm.double_kraus_to_ptm(np.diag([1, 1, d2, d2])) +\
            ptm.double_kraus_to_ptm(np.diag([0, 0, np.sqrt(1-d2**2),
                                             np.sqrt(1-d2**2)]))
        self.angle = angle
        self.two_ptm = p0 @ p1


class Measurement(Gate):

    def __init__(
            self,
            bit,
            time,
            sampler,
            output_bit=None,
            real_output_bit=None):
        """Create a Measurement gate. The measurement
        characteristics are defined by the sampler.
        The sampler is a coroutine object, which implements:

          declare, project, rel_prob = sampler.send((p0, p1))

        where `p0`, `p1` are two relative probabilities for the outcome 0 and
        1. `project` is the true post-measurement state of the system,
        `declare` is the declared outcome of the measurement.

        `rel_prob` is the conditional probability for the declaration, given
        the input and projection; for a perfect measurement this is 1.

        If sampler is None, a noiseless Monte Carlo sampler is instantiated
        with some random seed (depends on Numpy's defaults).

        After applying the circuit to a density matrix, the declared
        measurement results are stored in self.measurements.

        Additionally, the bits output_bit and real_output_bit (if defined)
        are set to the declared/projected value.

        See also: uniform_sampler, selection_sampler, uniform_noisy_sampler
        """

        super().__init__(time)
        self.is_measurement = True
        self.bit = bit
        self.label = r"$\circ\!\!\!\!\!\!\!\nearrow$"

        self.output_bit = output_bit
        if output_bit:
            self.involved_qubits.append(output_bit)
        self.real_output_bit = real_output_bit
        if real_output_bit:
            self.involved_qubits.append(real_output_bit)

        self.involved_qubits.append(bit)

        if sampler:
            self.sampler = sampler
        else:
            self.sampler = uniform_sampler()
        next(self.sampler)
        self.measurements = []
        self.probabilities = []
        self.projects = []

    def plot_gate(self, ax, coords):
        super().plot_gate(ax, coords)

        if self.output_bit:
            x = self.time
            y1 = coords[self.bit]
            y2 = coords[self.output_bit]

            ax.arrow(
                x,
                y1,
                0,
                y2 - y1 - 0.1,
                head_length=0.1,
                fc='w',
                width=0.2)

        if self.real_output_bit:
            x = self.time
            y1 = coords[self.bit]
            y2 = coords[self.real_output_bit]

            ax.arrow(
                x,
                y1,
                0,
                y2 - y1 - 0.1,
                head_length=0.1,
                fc='w',
                ec='k',
                ls=":")

    def apply_to(self, sdm):
        bit = self.bit
        p0, p1 = sdm.peak_measurement(bit)
        self.probabilities.append([p0, p1])

        declare, project, cond_prob = self.sampler.send((p0, p1))
        self.projects.append(project)
        self.measurements.append(declare)
        if self.output_bit:
            sdm.set_bit(self.output_bit, declare)
        sdm.project_measurement(bit, project)
        if self.real_output_bit:
            sdm.set_bit(self.real_output_bit, project)
        sdm.classical_probability *= cond_prob


class ResetGate(SinglePTMGate):

    def __init__(self, bit, time, population=0, *, state=None, **kwargs):
        if state is not None:
            warnings.warn('`state` keyword argument is deprecated,'
                          ' please use `population`', DeprecationWarning)
            population = state
        elif population > 0.5:
            state = 1
        else:
            state = 0

        p = ptm.gen_amp_damping_ptm(gamma_down=1-population,
                                    gamma_up=population)
        super().__init__(bit, time, p, **kwargs)
        self.state = state
        self.label = r"-> {}".format(state)


class ConditionalGate(Gate):

    def __init__(self, time, control_bit, zero_gates=[], one_gates=[]):
        """
        A container that applies gates depending on the state of a classical
        control bit.  The gates are applied in the order given.

        The times of the subgates are ignored, the gates are applied at the
        time of this gate.
        """

        super().__init__(time)

        self.control_bit = control_bit
        self.zero_gates = zero_gates
        self.one_gates = one_gates

        self.involved_qubits.append(control_bit)

        # enforce times (should not do anything, but just be sure)
        for g in self.zero_gates:
            g.time = self.time
        for g in self.one_gates:
            g.time = self.time

    def involves_qubit(self, bit):
        if bit == self.control_bit:
            return True

        return any(g.involves_qubit(bit)
                   for g in self.zero_gates + self.one_gates)

    def plot_gate(self, ax, coords):
        for g in self.zero_gates:
            g.plot_gate(ax, coords)
            x = self.time
            y = coords[g.involved_qubits[-1]]
            y2 = coords[self.control_bit]
            ax.plot((x, x), (y, y2), ".--", color='b')
        for g in self.one_gates:
            g.plot_gate(ax, coords)
            x = self.time
            y = coords[g.involved_qubits[-1]]
            y2 = coords[self.control_bit]
            ax.plot((x, x), (y, y2), ".--", color='r')

    def apply_to(self, sdm):
        sdm.ensure_classical(self.control_bit)
        if sdm.classical[self.control_bit] == 1:
            for g in self.one_gates:
                g.apply_to(sdm)
        else:
            for g in self.zero_gates:
                g.apply_to(sdm)


class ClassicalCNOT(Gate):

    def __init__(self, bit0, bit1, time, **kwargs):
        """A CNOT gate acting at time `time`, toggling bit1 if bit0 is 1.

        This gate enforces the bits to be classical, if you want a proper CNOT,
        build it using PTMs.
        """
        super().__init__(time, **kwargs)
        self.involved_qubits.append(bit0)
        self.involved_qubits.append(bit1)
        self.bit0 = bit0
        self.bit1 = bit1

    def plot_gate(self, ax, coords):
        ax.scatter((self.time,),
                   (coords[self.bit0],), color='k')
        ax.scatter((self.time,),
                   (coords[self.bit1],), color='k', marker=r'$\oplus$', s=70)

        xdata = (self.time, self.time)
        ydata = (coords[self.bit0], coords[self.bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)

    def apply_to(self, sdm):
        sdm.ensure_classical(self.bit0)
        sdm.ensure_classical(self.bit1)

        if sdm.classical[self.bit0] == 1:
            sdm.classical[self.bit1] = 1 - sdm.classical[self.bit1]


class ClassicalNOT(Gate):

    def __init__(self, bit, time, **kwargs):
        super().__init__(time, **kwargs)
        self.involved_qubits.append(bit)
        self.bit = bit
        self.label = "NOT"

    def apply_to(self, sdm):
        sdm.ensure_classical(self.bit)
        sdm.classical[self.bit] = 1 - sdm.classical[self.bit]


class Circuit:

    gate_classes = {"cphase": CPhase,
                    "hadamard": Hadamard,
                    "amp_ph_damping": AmpPhDamp,
                    "measurement": Measurement,
                    "rotate_y": RotateY,
                    "rotate_x": RotateX,
                    "rotate_z": RotateZ,
                    }

    def __init__(self, title="Unnamed circuit"):
        """Create an empty Circuit named `title`.
        """
        self.qubits = []
        self.gates = []
        self.title = title

    def get_qubit_names(self):
        """Return the names of all qubits in the circuit
        """
        return [qb.name for qb in self.qubits]

    def get_qubit(self, qubit_name):
        return [qb for qb in self.qubits if qb.name == qubit_name][0]

    def add_qubit(self, *args, **kwargs):
        """ Add a qubit. Either instantiate by hand

        qubit = Qubit("name", t1, t2)
        circ.add_qubit(qubit)

        or create the instance automatically:

        circ.add_qubit("name", t1, t2)
        """

        if isinstance(args[0], Qubit):
            qubit = args[0]
        else:
            qubit = Qubit(*args, **kwargs)

        qubit_names = [qb.name for qb in self.qubits]

        if qubit.name in qubit_names:
            raise ValueError(
                "Trying to add qubit with name {}:"
                " a qubit with this name already exists!".format(qubit.name))

        self.qubits.append(qubit)

        return self.qubits[-1]

    def add_gate(self, gate_type, *args, **kwargs):
        """Add a gate to the Circuit.

        gate_type can be a subclass of circuit.Gate, a string like "hadamard",
        or a gate class. in the latter two cases, an instance is
        created using args and kwargs
        """

        if isinstance(gate_type, type) and issubclass(gate_type, Gate):
            gate = gate_type(*args, **kwargs)
            self.add_gate(gate)
        elif isinstance(gate_type, str):
            gate = Circuit.gate_classes[gate_type](*args, **kwargs)
            self.gates.append(gate)
        elif isinstance(gate_type, Gate):
            self.gates.append(gate_type)
        elif gate_type is None:
            return
        else:
            raise(ValueError("Could not add gate: Gate not understood!"))

        return self.gates[-1]

    def add_subcircuit(self, subcircuit, time=0, name_map=None):
        """Add all gates of another circuit to this circuit.

        The qubit names in the subcircuit are mapped to the qubits in this
        circuit using `name_map`.

        name_map can be a dictionary, a list, or None.
        If it is a list, it the map is done according to the list
        subcircuit.gates.
        If it is None, no mapping takes place.

        All gate times in the subcircuit are shifted by `time`.
        """

        if not isinstance(name_map, dict):
            if isinstance(name_map, list):
                name_map = {
                    sg: g for sg,
                    g in zip(
                        subcircuit.get_qubit_names(),
                        name_map)}
            elif name_map is None:
                name_map = {g: g for g in subcircuit.get_qubit_names()}
            else:
                raise ValueError(
                    "name_map not understood. Pass a list, dict or None.")

        for g in subcircuit.gates:
            new_g = copy.copy(g)
            new_g.time += time
            new_g.involved_qubits = [name_map[b]
                                     for b in new_g.involved_qubits]

            self.add_gate(new_g)

    def __getattribute__(self, name):

        if name.find("add_") == 0:
            if name[4:] in Circuit.gate_classes:
                gate_type = Circuit.gate_classes[name[4:]]
                return functools.partial(self.add_gate, gate_type)

        return super().__getattribute__(name)

    def add_waiting_gates(self, tmin=None, tmax=None, only_qubits=None):
        """Add waiting gates to all qubits in the circuit.

        The waiting gates are determined by calling Qubit.make_idling_gate
        (AmpPhDamping by default).

        If only_qubits is an iterable containing qubit names, gates are only
        added to those qubits.

        The gates are added between all pairs of other gates between tmin and
        tmax. If tmin or tmax are not specified, they default to the time of
        the first (last) gate on any of the qubits in the circuit (or in
        only_qubits, if specified).

        tmax and tmin must either be None, a number, or a dictionary of numbers
        for each qubit.

        """
        all_gates = list(sorted(self.gates, key=lambda g: g.time_start))

        if not all_gates and (tmin is None or tmax is None):
            return

        if tmin is None:
            tmin = all_gates[0].time_start
        if tmax is None:
            tmax = all_gates[-1].time_end

        if not isinstance(tmin, dict):
            tmin = {qb.name: tmin for qb in self.qubits}
        if not isinstance(tmax, dict):
            tmax = {qb.name: tmax for qb in self.qubits}

        qubits_to_do = [qb for qb in self.qubits]
        if only_qubits:
            qubits_to_do = [qb for qb in qubits_to_do
                            if qb.name in only_qubits]

        for b in qubits_to_do:
            gts = [
                gate for gate in all_gates if gate.involves_qubit(
                    str(b)) and tmin[b.name] <= gate.time_start <= tmax[b.name]]

            if not gts:
                gate = b.make_idling_gate(tmin[b.name], tmax[b.name])
                if gate is not None:
                    gate.autogenerated = True
                    self.add_gate(gate)
            else:
                if (gts[0].time_start - tmin[b.name] > 1e-6 and not (
                        hasattr(gts[0], 'autogenerated') and
                        gts[0].autogenerated
                )):
                    gate = b.make_idling_gate(tmin[b.name], gts[0].time_start)
                    if gate is not None:
                        gate.autogenerated = True
                        self.add_gate(gate)
                if (tmax[b.name] - gts[-1].time_end > 1e-6 and not (
                    hasattr(gts[-1], 'autogenerated') and
                    gts[-1].autogenerated
                )):
                    gate = b.make_idling_gate(gts[-1].time_end, tmax[b.name])
                    if gate is not None:
                        gate.autogenerated = True
                        self.add_gate(gate)

                for g1, g2 in zip(gts[:-1], gts[1:]):
                    if (isinstance(g1, IdlingGate) or
                            isinstance(g2, IdlingGate)):
                        # there already is an idling gate,
                        # maybe added by hand, maybe from previous
                        # calls of this function, skip
                        pass
                    else:
                        gate = b.make_idling_gate(g1.time_end, g2.time_start)
                        if gate is not None:
                            gate.autogenerated = True
                            self.add_gate(gate)

    def order(self, toposort=True):
        """ Reorder the gates in the circuit so that they are applied in
        temporal order. If `toposort` is ``True`` and any freedom exists
        when choosing the order of commuting gates, the order is chosen so
        that measurement gates are applied "as soon as possible"; this means
        that when applying to a SparseDM, the measured qubits can be removed,
        which reduces computational cost.

        This function should always be called after defining the circuit and
        before applying it.

        See also: :func:`Circuit.apply_to`.

        Parameters
        ----------
        toposort: bool
            Whether to apply topological sorting routine (that may be costly)
            or to imply only chronological sorting. Defaults to ``True``.
            See :func:`tp.partial_greedy_toposort` for details.
        """
        if not toposort:
            self.gates = sorted(self.gates, key=lambda g: g.time)
            return

        all_gates = list(enumerate(sorted(self.gates, key=lambda g: g.time)))

        gts_list = []
        targets = []
        for n, b in enumerate(self.qubits):
            gts = [n for n, gate in all_gates if gate.involves_qubit(str(b))]
            if any(all_gates[g][1].is_measurement and
                   all_gates[g][1].involved_qubits[-1] == b.name
                   for g in gts):
                targets.append(n)
            gts_list.append(gts)

        order = tp.partial_greedy_toposort(gts_list, targets=targets)

        for n, i in enumerate(order):
            all_gates[i][1].annotation = "%d" % n

        new_order = []
        for i in order:
            new_order.append(all_gates[i][1])

        self.gates = new_order

    def apply_to(self, sdm, apply_all_pending=True):
        """Apply the gates in the Circuit to a sparsedm.SparseDM density
        matrix.  The gates are applied in the order given in self.gates, which
        is the order in which they are added to the Circuit. To reorder them to
        reflect the temporal order, call self.order()

        See also: Circuit.order()
        """
        for gate in self.gates:
            gate.apply_to(sdm)

        if apply_all_pending:
            sdm.apply_all_pending()

    def plot(self, show_annotations=False):
        """
        Plot the circuit using matplotlib.

        returns
            figure : matplotlib figure object
            ax     : matplotlib axis object
        """
        times = [g.time for g in self.gates]

        tmin = min(times)
        tmax = max(times)

        if tmax - tmin < 0.1:
            tmin -= 0.05
            tmax += 0.05

        buffer = (tmax - tmin) * 0.05

        coords = {str(qb): number for number, qb in enumerate(self.qubits)}

        figure = plt.gcf()

        ax = figure.add_subplot(1, 1, 1, frameon=True)

        ax.set_title(self.title, loc="left")
        ax.get_yaxis().set_ticks([])

        ax.set_xlim(tmin - 5 * buffer, tmax + 3 * buffer)
        ax.set_ylim(-1, len(self.qubits))

        ax.set_xlabel('time')

        self._plot_qubit_lines(ax, coords, tmin, tmax)

        for gate in self.gates:
            gate.plot_gate(ax, coords)
            if show_annotations:
                gate.annotate_gate(ax, coords)
        return figure, ax

    def _plot_qubit_lines(self, ax, coords, tmin, tmax):
        buffer = (tmax - tmin) * 0.05
        xdata = (tmin - buffer, tmax + buffer)
        for qubit in coords:
            ydata = (coords[qubit], coords[qubit])
            line = mp.lines.Line2D(xdata, ydata, color='k')
            ax.add_line(line)
            ax.text(
                xdata[0] - 2 * buffer,
                ydata[0],
                str(qubit),
                color='k',
                ha='center',
                va='center')

    def make_full_PTM(self, qubit_order, i_really_want_to_do_this=False):
        '''
        Generates the PTM of the entire circuit, assuming no measurements.
        Warning - this is a very badly scaling process, and is currently
        only performed on a CPU.
        Assumes that the circuit has been ordered!
        '''
        num_qubits = len(self.qubits)
        if qubit_order is not None:
            qubits = qubit_order
        else:
            qubits = [q.name for q in self.qubits]
        if num_qubits > 5 and i_really_want_to_do_this is False:
            raise ValueError('I dont think you want to do this')
        full_PTM = np.identity(4**(num_qubits)).reshape((4, 4)*num_qubits)

        for gate in self.gates:
            if gate.is_measurement:
                raise TypeError('Cannot get the PTM of a measurement')
            if gate.conditional_bit:
                raise TypeError('Cannot get the PTM with a conditional gate')

            if len(gate.involved_qubits) == 1:
                # Single-qubit gate
                bit = qubits.index(gate.involved_qubits[0])
                dummy_idx = num_qubits*2
                in_indices = list(reversed(range(num_qubits*2)))
                out_indices = list(reversed(range(num_qubits*2)))
                in_indices[2*num_qubits - bit - 1] = dummy_idx
                ptm_indices = [bit, dummy_idx]
                single_ptm = ptm.to_0xyz_basis(gate.ptm)
                full_PTM = np.einsum(single_ptm, ptm_indices, full_PTM,
                                     in_indices, out_indices, optimize=True)

            elif len(gate.involved_qubits) == 2:
                # Two qubit gate
                bit0 = qubits.index(gate.involved_qubits[0])
                bit1 = qubits.index(gate.involved_qubits[1])

                two_ptm = ptm.to_0xyz_basis(gate.two_ptm).reshape((4, 4, 4, 4))
                dummy_idx0, dummy_idx1 = 2*num_qubits, 2*num_qubits + 1
                out_indices = list(reversed(range(2*num_qubits)))
                in_indices = list(reversed(range(2*num_qubits)))
                in_indices[num_qubits*2 - bit0 - 1] = dummy_idx0
                in_indices[num_qubits*2 - bit1 - 1] = dummy_idx1
                two_ptm_indices = [
                    bit1, bit0,
                    dummy_idx1, dummy_idx0
                ]
                full_PTM = np.einsum(
                    two_ptm, two_ptm_indices, full_PTM,
                    in_indices, out_indices, optimize=True)

            else:
                raise ValueError(
                    'Sorry, feature not implemented for >2 qubits')

        full_PTM = full_PTM.reshape(4**num_qubits, 4**num_qubits).T
        return full_PTM


def selection_sampler(result=0):
    """ A sampler always returning the measurement result `result`, and not
    making any measurement errors. Useful for testing or state preparation.

    See also: Measurement
    """
    while True:
        yield result, result, 1


def uniform_sampler(rng=None, *, state=None, seed=None):
    """A sampler using natural Monte Carlo sampling, and always declaring the
    correct result. The stream of measurement results is defined by the seed;
    you should never use two samplers with the same seed in one circuit.

    See also: Measurement
    """
    if state:
        warnings.warn('`state` keyword argument is deprecated,'
                      ' please use `rng`', DeprecationWarning)
        rng = state
    if seed:
        warnings.warn('`seed` keyword argument is deprecated,'
                      ' please use `rng`', DeprecationWarning)
        rng = seed
    rng = _ensure_rng(rng)
    primers_nones = yield
    while not primers_nones:
        primers_nones = yield
    p0, p1 = primers_nones
    while True:
        r = rng.random_sample()
        if r < p0 / (p0 + p1):
            p0, p1 = yield 0, 0, 1
        else:
            p0, p1 = yield 1, 1, 1


def uniform_noisy_sampler(readout_error, rng=None, *, state=None, seed=None):
    """A sampler using natural Monte Carlo sampling and including the
    possibility of declaring the wrong measurement result with probability
    `readout_error` (now allows asymmetry)

    See also: Measurement
    """
    if state:
        warnings.warn('`state` keyword argument is deprecated,'
                      ' please use `rng`', DeprecationWarning)
        rng = state
    if seed:
        warnings.warn('`seed` keyword argument is deprecated,'
                      ' please use `rng`', DeprecationWarning)
        rng = seed
    rng = _ensure_rng(rng)
    if not type(readout_error) in [list, tuple]:
        readout_error = [readout_error, readout_error]
    primers_nones = yield
    while not primers_nones:
        primers_nones = yield
    p0, p1 = primers_nones
    while True:
        r = rng.random_sample()
        if r < p0 / (p0 + p1):
            proj = 0
        else:
            proj = 1
        r = rng.random_sample()
        if r < readout_error[proj]:
            decl = 1 - proj
            prob = readout_error[proj]
        else:
            decl = proj
            prob = 1 - readout_error[proj]

        ps = yield decl, proj, prob
        while not ps:
            ps = yield
        p0, p1 = ps


class BiasedSampler:
    '''A sampler that returns a uniform choice but with probabilities weighted
    as p_twiddle=p^alpha/Z, with Z a normalisation constant. Also allows for
    readout error to be input when the sampling is called.

    All the class does is to store the product of all p_twiddles for
    renormalisation purposes
    '''

    def __init__(self, readout_error, alpha, rng=None,
                 *, state=None, seed=None):
        '''
        @alpha: number between 0 and 1 for renormalisation purposes.
        '''
        if state:
            warnings.warn('`state` keyword argument is deprecated,'
                          ' please use `rng`', DeprecationWarning)
            rng = state
        if seed:
            warnings.warn('`seed` keyword argument is deprecated,'
                          ' please use `rng`', DeprecationWarning)
            rng = seed
        self.alpha = alpha
        self.p_twiddle = 1
        self.rng = _ensure_rng(rng)
        self.readout_error = readout_error
        ro_temp = readout_error ** self.alpha
        self.ro_renormalized = ro_temp / \
            (ro_temp + (1 - readout_error)**self.alpha)

    def __next__(self):
        pass

    def send(self, ps):
        '''
        @readout_error: probability of the state update and classical output
        disagreeing
        '''

        if ps is None:
            return None

        p0, p1 = ps

        # renormalise probability values
        p0_temp = (p0 / (p0 + p1))**self.alpha
        p1_temp = (p1 / (p0 + p1))**self.alpha
        p0_renormalized = p0_temp / (p0_temp + p1_temp)

        r = self.rng.random_sample()
        if r < p0_renormalized:
            proj = 0
            self.p_twiddle = self.p_twiddle * p0_renormalized
        else:
            proj = 1
            self.p_twiddle = self.p_twiddle * (1 - p0_renormalized)
        r = self.rng.random_sample()
        if r < self.ro_renormalized:
            decl = 1 - proj
            prob = self.readout_error
            self.p_twiddle = self.p_twiddle * self.ro_renormalized
        else:
            decl = proj
            prob = 1 - self.readout_error
            self.p_twiddle = self.p_twiddle * (1 - self.ro_renormalized)
        return decl, proj, prob


def _ensure_rng(rng):
    """Takes random number generator (RNG) or seed as input and instantiates
    and returns RNG, initialized by seed, if it is provided.
    """
    if not hasattr(rng, 'random_sample'):
        if not rng:
            warnings.warn('No random number generator (or seed) provided, '
                          'computation will not be reproducible.')
        # Assuming that we have seed provided instead of RNG
        rng = np.random.RandomState(seed=rng)
    return rng
