import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np

import tp

import functools

class Qubit:

    def __init__(self, name, t1=np.inf, t2=np.inf):
        self.name = name
        self.t1 = max(t1, 1e-10)
        self.t2 = max(t2, 1e-10)

    def __str__(self):
        return self.name

class Gate:
    def __init__(self, time, conditional_bit=None):
        self.is_measurement = False
        self.time = time
        self.label = r"$G"
        self.involved_qubits = []
        self.annotation = None
        self.conditional_bit = conditional_bit
        if self.conditional_bit:
            self.involved_qubits.append(self.conditional_bit)

    def plot_gate(self, ax, coords):
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
            ax.plot( (x,x), (y,y2), ".--", color='k')

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
        if self.conditional_bit is not None:
            sdm.ensure_classical(self.conditional_bit)
            if sdm.classical[self.conditional_bit] == 1:
                f = sdm.__getattribute__(self.method_name)
                f(*self.involved_qubits[1:], **self.method_params)

        else:
            f = sdm.__getattribute__(self.method_name)
            f(*self.involved_qubits, **self.method_params)

class Hadamard(Gate):
    def __init__(self, bit, time, **kwargs):
        super().__init__(time, **kwargs)
        self.involved_qubits.append(bit)
        self.label = r"$H$"
        self.method_name = "hadamard"
        self.method_params = {}

class RotateY(Gate):
    def __init__(self, bit, time, angle, **kwargs):
        super().__init__(time, **kwargs)
        self.involved_qubits.append(bit)

        multiple_of_pi = angle/np.pi
        if np.allclose(multiple_of_pi, 1):
            self.label = r"$\pi$"
        elif not np.allclose(angle, 0) and np.allclose(np.round(1/multiple_of_pi, 0), 1/multiple_of_pi):
                divisor = 1/multiple_of_pi
                self.label = r"$\pi/%d$"%divisor
        else:
            self.label = r"$R_y(%g)$"%angle

        self.method_name = "rotate_y"
        self.method_params = {"angle": angle}

class CPhase(Gate):
    def __init__(self, bit0, bit1, time, **kwargs):
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

class AmpPhDamp(Gate):
    def __init__(self, bit, time, duration, t1, t2, **kwargs):
        super().__init__(time, **kwargs)
        self.involved_qubits.append(bit)
        self.duration = duration
        self.t1 = t1
        self.t2 = t2
        self.method_name = "amp_ph_damping"
        self.method_params = {"gamma": 1 - np.exp(-duration/t1),
                "lamda": 1 - np.exp(-duration/t2) }

    def plot_gate(self, ax, coords):
        ax.scatter((self.time),
                   (coords[self.involved_qubits[-1]]), color='k', marker='x')
        ax.annotate(
            r"$%g\,\mathrm{ns}$" %
            self.duration, (self.time, coords[
                self.involved_qubits[0]]), xytext=(
                0, 20), textcoords='offset points', ha='center')

class Measurement(Gate):
    def __init__(self, bit, time, sampler, output_bit=None, real_output_bit=None):
        """Create a Measurement gate. The measurement 
        characteristics are defined by the sampler.
        The sampler is a coroutine object, which implements:

          declare, project, rel_prob = sampler.send(p0, p1)

        where p0, p1 are two relative probabilities for the outcome 0 and 1.
        project is the true post-measurement state of the system,
        while declare is the declared outcome of the measurement.

        rel_prob is the conditional probability for the declaration, given the 
        input and projection; for a perfect measurement this is 1.

        If sampler is None, a perfect natural Monte Carlo sampler is instantiated.

        After applying the circuit to a density matrix, the declared measurement results
        are stored in self.measurements.

        Additionally, the bits output_bit and real_output_bit (if defined) 
        are set to the declared/projected value.
        """

        super().__init__(time)
        self.is_measurement = True
        self.involved_qubits.append(bit)
        self.bit = bit
        self.label = r"$\circ\!\!\!\!\!\!\!\nearrow$"


        
        self.output_bit = output_bit
        if output_bit:
            self.involved_qubits.append(output_bit)
        self.real_output_bit = real_output_bit
        if real_output_bit:
            self.involved_qubits.append(real_output_bit)

        if sampler:
            self.sampler = sampler
        else:
            self.sampler = uniform_sampler()
        next(self.sampler)
        self.measurements = []

    def plot_gate(self, ax, coords):
        super().plot_gate(ax, coords)

        if self.output_bit:
            x = self.time
            y1 = coords[self.bit]
            y2 = coords[self.output_bit]

            ax.arrow(x, y1, 0, y2-y1-0.1, head_length=0.1, fc='w')

        if self.real_output_bit:
            x = self.time
            y1 = coords[self.bit]
            y2 = coords[self.real_output_bit]

            ax.arrow(x, y1, 0, y2-y1-0.1, head_length=0.1, fc='w', ec='k', ls=":")

    def apply_to(self, sdm):
        bit = self.involved_qubits[0]
        p0, p1 = sdm.peak_measurement(bit)

        declare, project, cond_prob = self.sampler.send((p0, p1))

        self.measurements.append(declare)
        if self.output_bit:
            sdm.set_bit(self.output_bit, declare)
        sdm.project_measurement(bit, project)
        if self.real_output_bit:
            sdm.set_bit(self.real_output_bit, project)
        sdm.classical_probability *= cond_prob

class Circuit:

    gate_classes = {"cphase": CPhase, 
            "hadamard": Hadamard,
            "amp_ph_damping": AmpPhDamp,
            "measurement" : Measurement,
            "rotate_y": RotateY,
            }

    def __init__(self, title="Unnamed circuit"):
        self.qubits = []
        self.gates = []
        self.title = title

    def get_qubit_names(self):
        return [qb.name for qb in self.qubits]

    def add_qubit(self, *args, **kwargs):
        """ Add a qubit. Either 

        qubit = Qubit("name", t1, t2)
        circ.add_qubit(qubit)

        or create the instance automatically:

        circ.add_qubit("name", t1, t2)
        """

        if isinstance(args[0], Qubit):
            qubit = args[0]
            self.qubits.append(qubit)
        else:
            qb = Qubit(*args, **kwargs)
            self.qubits.append(qb)

        return self.qubits[-1]

    def add_gate(self, gate_type, *args, **kwargs):
        """Add a gate to the Circuit.

        gate_type can be circuit.Gate, a string like "hadamard",
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

        return self.gates[-1]

    def __getattribute__(self, name):

        if name.find("add_") == 0:
            if name[4:] in Circuit.gate_classes:
                gate_type = Circuit.gate_classes[name[4:]]
                return functools.partial(self.add_gate, gate_type)

        return super().__getattribute__(name)

    def add_waiting_gates(self, tmin=None, tmax=None, qubits=None):

        all_gates = list(sorted(self.gates, key=lambda g: g.time))



        if not all_gates and (tmin is None or tmax is None):
            return
        
        if tmin is None:
            tmin = all_gates[0].time
        if tmax is None:
            tmax = all_gates[-1].time

        qubits_to_do = self.qubits
        if qubits:
            qubits_to_do = [qb for qb in qubits_to_do if qb.name in qubits]

        for b in qubits_to_do:
            gts = [gate for gate in all_gates if gate.involves_qubit(str(b))
                    and tmin <= gate.time <= tmax]

            if not gts:
                self.add_gate(
                    AmpPhDamp(
                        str(b),
                        (tmax + tmin) / 2,
                        tmax - tmin, b.t1, b.t2))
            else:
                if gts[0].time - tmin > 1e-6:
                    self.add_gate(
                        AmpPhDamp(
                            str(b),
                            (gts[0].time + tmin) / 2,
                            gts[0].time - tmin, b.t1, b.t2))
                if tmax - gts[-1].time > 1e-6:
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

    def plot(self, show_annotations=False):
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

def selection_sampler(result=0):
    while True:
        yield result, result, 1

def uniform_sampler(seed=42):
    rng = np.random.RandomState(seed)
    p0, p1 = yield
    while True:
        r = rng.random_sample()
        if r < p0/(p0+p1):
            p0, p1 = yield 0, 0, 1
        else:
            p0, p1 = yield 1, 1, 1

def uniform_noisy_sampler(readout_error, seed=42):
    rng = np.random.RandomState(seed)
    p0, p1 = yield
    while True:
        r = rng.random_sample()
        if r < p0/(p0+p1):
            proj = 0
        else:
            proj = 1
        r = rng.random_sample()
        if r < readout_error:
            decl = 1 - proj
            prob = readout_error
        else:
            decl = proj
            prob = 1 - readout_error
        p0, p1 = yield proj, decl, prob 
