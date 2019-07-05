import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_golden_mean = (np.sqrt(5) - 1.0) / 2.0


def plot(circuit, *, ax=None, realistic_timing=True, qubit_order=None):
    plotter = MatplotlibPlotter(
        circuit, ax, None, qubit_order=qubit_order,
        realistic_timing=realistic_timing, )
    return plotter.plot()


class MatplotlibPlotter:
    def __init__(self, circuit, ax=None, params=None, qubit_order=None,
                 realistic_timing=True):
        self.circuit = circuit
        self.realistic_timing = realistic_timing

        if ax is not None:
            self.fig = None
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots()

        self.params = {
            'linewidth': 1,
            'edgecolor': 'black',
            'facecolor': 'white'
        }
        if params is not None:
            self.params.update(params)

        if callable(qubit_order):
            self.qubits = sorted(circuit.qubits, key=qubit_order)
        elif hasattr(qubit_order, '__iter__'):
            self.qubits = tuple(qubit_order)
        elif qubit_order is None:
            self.qubits = circuit.qubits
        else:
            raise ValueError('Qubit order must be a list, callable or None')

    def plot(self):
        for gate in self.circuit.gates:
            self.plot_gate(gate)

        return self.fig

    def plot_gate(self, gate):
        """

        Parameters
        ----------
        gate : quantumsim.circuits.Gate

        Returns
        -------

        """
        style = gate.plot_metadata.get('style')
        if style == 'box':
            label = gate.plot_metadata.get('label')
            qubits = [self.qubits.index(qubit) for qubit in gate.qubits]
            return self.plot_box_with_label(
                gate.time_start, gate.time_end, min(qubits), max(qubits),
                label, self.params)

    def plot_box_with_label(self, time_start, time_end,
                            n_qubit_start, n_qubit_end, label,
                            extra_props=None):
        """
        Parameters
        ----------
        time_start : float
        time_end : float
        n_qubit_start : int
        n_qubit_end : int
        label : str or None
        extra_props : dict or None
        """
        if extra_props is not None:
            box_props = self.params.copy()
            box_props.update(extra_props)
        else:
            box_props = self.params

        box_y = n_qubit_start - 0.5 * _golden_mean
        box_dy = n_qubit_end - n_qubit_start + 0.5*_golden_mean
        box_x = time_start
        box_dx = time_end - time_start

        rect = Rectangle((box_x, box_y), box_dx, box_dy, **box_props)
        self.ax.add_patch(rect)
        if label is not None:
            self.ax.text(box_x + 0.5 * box_dx, box_y + 0.5 * box_dy,
                         label, ha='center', va='center')
