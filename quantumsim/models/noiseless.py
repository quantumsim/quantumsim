'''
Models for gates in the absence of noise.
Copied mostly from circuit.py
'''

from quantumsim import ptm
import matplotlib as mp
from quantumsim.circuit import (
    Gate, SinglePTMGate, TwoPTMGate, _format_angle,
    uniform_sampler)
import numpy as np


class RotateX(SinglePTMGate):

    def __init__(
            self,
            bit,
            time,
            angle,
            **kwargs):
        """ A rotation around the x-axis on the bloch sphere by `angle`.
        """
        p = ptm.rotate_x_ptm(angle)
        super().__init__(bit, time, p, **kwargs)
        self.adjust(angle)

    def set_labels(self, angle):
        self.angle = angle
        self.label = r"$R_x({})$".format(_format_angle(angle))

    def adjust(self, angle):
        p = ptm.rotate_x_ptm(angle)
        self.ptm = p
        self.set_labels(angle)

class XGate(SinglePTMGate):

    def __init__(
            self,
            bit,
            time,
            **kwargs):
        """ A rotation around the x-axis on the bloch sphere by pi.
        """
        p = ptm.rotate_x_ptm(-np.pi)
        super().__init__(bit, time, p, **kwargs)

    def set_labels(self):
        self.label = r"$X$"

class YGate(SinglePTMGate):

    def __init__(
            self,
            bit,
            time,
            **kwargs):
        """ A rotation around the x-axis on the bloch sphere by `angle`.
        """
        p = ptm.rotate_x_ptm(-np.pi)
        super().__init__(bit, time, p, **kwargs)

    def set_labels(self):
        self.label = r"$Y$"


class ZGate(SinglePTMGate):

    def __init__(
            self,
            bit,
            time,
            **kwargs):
        """ A rotation around the x-axis on the bloch sphere by `angle`.
        """
        p = ptm.rotate_x_ptm(-np.pi)
        super().__init__(bit, time, p, **kwargs)

    def set_labels(self):
        self.label = r"$Z$"


class RotateY(SinglePTMGate):

    def __init__(
            self,
            bit,
            time,
            angle,
            **kwargs):
        """ A rotation around the y-axis on the bloch sphere by `angle`.
        """
        p = ptm.rotate_y_ptm(angle)
        super().__init__(bit, time, p, **kwargs)
        self.adjust(angle)

    def set_labels(self, angle):
        self.angle = angle
        self.label = r"$R_y({})$".format(_format_angle(angle))

    def adjust(self, angle):
        p = ptm.rotate_y_ptm(angle)
        self.ptm = p
        self.set_labels(angle)


class RotateZ(SinglePTMGate):

    def __init__(self, bit, time, angle, **kwargs):
        """ A rotation around the z-axis on the bloch sphere by `angle`.
        """
        p = ptm.rotate_z_ptm(angle)
        super().__init__(bit, time, p, **kwargs)
        self.adjust(angle)

    def set_labels(self, angle):
        self.angle = angle
        self.label = r"$R_z({})$".format(_format_angle(angle))

    def adjust(self, angle):
        p = ptm.rotate_z_ptm(angle)
        self.ptm = p
        self.set_labels(angle)


class Hadamard(SinglePTMGate):

    def __init__(self, bit, time, **kwargs):
        """A Hadamard gate on qubit `bit` acting at a point in time `time`
        """
        super().__init__(bit, time, ptm.hadamard_ptm(), **kwargs)
        self.label = r"$H$"


class CPhase(TwoPTMGate):

    def __init__(self, bit0, bit1, time, **kwargs):
        """A CNOT gate acting at time `time` between bit0 and bit1
        (bit1 is the control bit).
        """
        kraus = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ])

        p = ptm.double_kraus_to_ptm(kraus)
        super().__init__(bit0, bit1, p, time, **kwargs)


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


class ISwap(TwoPTMGate):

    def __init__(self, bit0, bit1, time, **kwargs):
        """
        ISwap gate, described by the two qubit operator

        1  0 0 0
        0  0 i 0
        0  i 0 0
        0  0 0 1
        """
        kraus0 = np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ])

        p1 = ptm.double_kraus_to_ptm(kraus0)

        super().__init__(bit0, bit1, p1, time, **kwargs)

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

        self.sampler = sampler
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
