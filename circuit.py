import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np

import tp


class Circuit:

    def __init__(self):
        self.qubits = []
        self.gates = []

    def add_qubit(self, name):
        self.qubits.append(name)

    def add_gate(self, gate):
        self.gates.append(gate)

    def add_waiting_gates(self):
        all_gates = list(sorted(self.gates, key=lambda g: g.time))

        tmin = all_gates[0].time
        tmax = all_gates[-1].time

        for b in self.qubits:
            gts = [gate for gate in all_gates if gate.involves_qubit(str(b))]

            if abs(tmin - gts[0].time) > 1e-6:
                self.add_gate(
                    AmpPhDamp(
                        str(b),
                        (gts[0].time + tmin) / 2,
                        gts[0].time - tmin,
                    b.t1, b.t2))
            if abs(tmax - gts[-1].time) > 1e-6:
                self.add_gate(AmpPhDamp(
                    str(b), (gts[-1].time + tmax) / 2, tmax - gts[-1].time,
                    b.t1, b.t2))

            for g1, g2 in zip(gts[:-1], gts[1:]):
                self.add_gate(
                    AmpPhDamp(
                        str(b),
                        (g1.time + g2.time) / 2,
                        g2.time - g1.time,
                    b.t1, b.t2))

    def order(self):
        all_gates = list(enumerate(sorted(self.gates, key=lambda g: g.time)))
        measurements = [n for n, gate in all_gates if gate.is_measurement]
        dependencies = {n: set() for n, gate in all_gates}

        for b in self.qubits:
            gts = [n for n, gate in all_gates if gate.involves_qubit(str(b))]
            for g1, g2 in zip(gts[:-1], gts[1:]):
                dependencies[g2] |= {g1}

        order = tp.greedy_toposort(dependencies, set(measurements))

        for n, i in enumerate(order):
            all_gates[i][1].annotation = "%d" % n

        new_order = []
        for i in order:
            new_order.append(all_gates[i][1])

        self.gates = new_order

    def apply_to(self, sdm):
        for gate in self.gates:
            gate.apply_to(sdm)

    def plot(self):
        times = [g.time for g in self.gates]

        tmin = min(times)
        tmax = max(times)

        buffer = (tmax - tmin) * 0.05

        coords = {str(qb): number for number, qb in enumerate(self.qubits)}

        figure = plt.figure(
            facecolor='w',
            edgecolor='w'
        )

        ax = figure.add_subplot(1, 1, 1, frameon=True)
        # ax.set_axis_off()

        ax.set_xlim(tmin - 5 * buffer, tmax + 3 * buffer)
        ax.set_ylim(-1, len(self.qubits) + 1)

        self._plot_qubit_lines(ax, coords, tmin, tmax)

        for gate in self.gates:
            gate.plot_gate(ax, coords)
            gate.annotate_gate(ax, coords)

        plt.show()

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


class Qubit:

    def __init__(self, name, t1, t2):
        self.name = name
        self.t1 = max(t1, 1e-10)
        self.t2 = max(t2, 1e-10)

    def __str__(self):
        return self.name

class Gate:
    def __init__(self, time):
        self.is_measurement = False
        self.time = time
        self.label = r"$G"
        self.involved_qubits = []
        self.annotation = None

    def plot_gate(self, ax, coords):
        x = self.time
        y = coords[self.involved_qubits[0]]
        ax.text(
            x, y, self.label,
            color='k',
            ha='center',
            va='center',
            bbox=dict(ec='k', fc='w', fill=True),
        )

    def annotate_gate(self, ax, coords):
        if self.annotation:
            x = self.time
            y = coords[self.involved_qubits[0]]
            ax.annotate(self.annotation, (x, y),
                        color='r',
                        xytext=(0, -15), textcoords='offset points', ha='center')

    def involves_qubit(self, bit):
        return bit in self.involved_qubits

    def apply_to(self, sdm):
        f = sdm.__getattribute__(self.method_name)

        f(*self.involved_qubits, **self.method_params)

class Hadamard(Gate):
    def __init__(self, bit, time):
        super().__init__(time)
        self.involved_qubits.append(bit)
        self.label = r"$H$"
        self.method_name = "hadamard"
        self.method_params = {}

class CPhase(Gate):
    def __init__(self, bit0, bit1, time):
        super().__init__(time)
        self.involved_qubits.append(bit0)
        self.involved_qubits.append(bit1)
        self.method_name = "cphase"
        self.method_params = {}

    def plot_gate(self, ax, coords):
        bit0 = self.involved_qubits[0]
        bit1 = self.involved_qubits[1]
        ax.scatter((self.time, self.time),
                   (coords[bit0], coords[bit1]), color='k')

        xdata = (self.time, self.time)
        ydata = (coords[bit0], coords[bit1])
        line = mp.lines.Line2D(xdata, ydata, color='k')
        ax.add_line(line)

class AmpPhDamp(Gate):
    def __init__(self, bit, time, duration, t1, t2):
        super().__init__(time)
        self.involved_qubits.append(bit)
        self.duration = duration
        self.t1 = t1
        self.t2 = t2
        self.method_name = "amp_ph_damping"
        self.method_params = {"gamma": 1 - np.exp(-duration/t1),
                "lamda": 1 - np.exp(-duration/t2) }

    def plot_gate(self, ax, coords):
        ax.scatter((self.time),
                   (coords[self.involved_qubits[0]]), color='k', marker='x')
        ax.annotate(
            r"$%g\,\mathrm{ns}$" %
            self.duration, (self.time, coords[
                self.involved_qubits[0]]), xytext=(
                0, 3), textcoords='offset points', ha='center')

class Measurement(Gate):
    def __init__(self, bit, time, sampler):
        super().__init__(time)
        self.is_measurement = True
        self.involved_qubits.append(bit)
        self.label = r"$\circ\!\!\!\!\!\!\!\nearrow$"

        self.sampler = sampler


        self.measurements = []


    def apply_to(self, sdm):
        bit = self.involved_qubits[0]
        p0, p1 = sdm.peak_measurement(bit)

        r = np.random.random()
        if r < p0/(p0+p1):
            self.measurements.append(0)
            sdm.project_measurement(bit, 0)
        else:
            self.measurements.append(1)
            sdm.project_measurement(bit, 1)


if __name__ == "__main__":
    cp = CPhase(0, 1, 0)
    h = Hadamard(0, 2)

    ct = Circuit()
    bits = ["D1", "A1", "D2", "A2", "D3"]
    for qb in bits:
        ct.add_qubit(Qubit(qb, 10, 10))

    ct.add_gate(CPhase("A1", "D1", 0))
    ct.add_gate(CPhase("A1", "D2", 100))
    ct.add_gate(CPhase("A2", "D2", 0))
    ct.add_gate(CPhase("A2", "D3", 100))

    ct.add_gate(Hadamard("A1", -50))
    ct.add_gate(Hadamard("A1", 200))
    ct.add_gate(Hadamard("A2", -50))
    ct.add_gate(Hadamard("A2", 200))

    ct.add_gate(Measurement("A1", 300))
    ct.add_gate(Measurement("A2", 300))

    ct.add_waiting_gates()

    ct.order()

    ct.plot()

