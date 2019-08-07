import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from matplotlib.patches import Rectangle

_golden_mean = (np.sqrt(5) - 1.0) / 2.0


def plot(circuit, *, ax=None, qubit_order=None, gate_offset=2.):
    plotter = MatplotlibPlotter(
        circuit, ax, qubit_order, gate_offset)
    return plotter.plot()


class MatplotlibPlotter:
    zorders = {
        'line': 1,
        'marker': 1,
        'box': 10,
        'text': 20,
    }

    def __init__(self, circuit, ax, qubit_order, gate_offset, figsize=None):
        self.circuit = circuit
        self.gate_offset = 0.5*gate_offset

        if ax is not None:
            self.fig = None
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(
                figsize=figsize or (7, 0.5*len(self.circuit.qubits)))
            self.ax.set_ylim(-1, len(self.circuit.qubits))
            self.ax.set_yticks([])
            tmin = self.circuit.time_start
            tmax = self.circuit.time_end

            if tmax - tmin < 0.1:
                tmin -= 0.05
                tmax += 0.05
            buffer = (tmax - tmin) * 0.05
            self.ax.set_xlim(tmin - 2.5 * buffer, tmax + 1.5 * buffer)

        if callable(qubit_order):
            self.qubits = sorted(circuit.qubits, key=qubit_order)
        elif hasattr(qubit_order, '__iter__'):
            self.qubits = tuple(qubit_order)
        elif qubit_order is None:
            self.qubits = circuit.qubits
        else:
            raise ValueError('Qubit order must be a list, callable or None')

    def plot(self):
        for qubit in self.circuit.qubits:
            self._plot_qubit_line(qubit)
            self._anotate_qubit(qubit)
        for gate in self.circuit.gates:
            self._plot_gate(gate)
        return self.fig

    def _plot_single_qubit_marker(self, qubit, time_start, duration,
                                  marker_dict):
        if marker_dict is None:
            return
        if not isinstance(marker_dict, dict):
            raise RuntimeError('marker_dict must be dict')
        style = marker_dict.pop('style', 'marker')
        n = self._qubit_number(qubit)
        if style == 'marker':
            time = time_start + 0.5 * duration
            marker_kwargs = self._get_marker_kwargs(marker_dict)
            self.ax.scatter((time,), (self._qubit_number(qubit),),
                            **marker_kwargs)
        elif style == 'box':
            return self._plot_box_with_label(
                time_start, time_start + duration, n, n, marker_dict)
        else:
            raise RuntimeError('Unknown marker style: {}'.format(style))

    def _plot_gate(self, gate):
        """

        Parameters
        ----------
        gate : quantumsim.circuits.Gate

        Returns
        -------

        """
        metadata = deepcopy(gate.plot_metadata)
        # By default we will plot a box
        style = metadata.pop('style', 'box')

        if style == 'box':
            return self._plot_box_with_label(
                gate.time_start, gate.time_end, *self._qubit_range(gate.qubits),
                metadata)
        elif style == 'line':
            time = gate.time_start + 0.5 * gate.duration
            markers = metadata.pop('markers')
            self._plot_vline(time, *self._qubit_range(gate.qubits), metadata)
            if markers is not None:
                for qubit, marker in zip(gate.qubits, markers):
                    self._plot_single_qubit_marker(
                        qubit, gate.time_start, gate.duration, marker)
        elif style == 'marker':
            for qubit in gate.qubits:
                self._plot_single_qubit_marker(
                    qubit, gate.time_start, gate.duration, metadata)
        else:
            raise RuntimeError("Unknown gate plotting style: {}".format(style))

    def _anotate_qubit(self, qubit):
        xlim = self.ax.get_xlim()
        xlim = xlim[1] - xlim[0]
        time = self.circuit.time_start - 0.01*xlim
        self.ax.text(time, self._qubit_number(qubit), qubit,
                     ha='right', va='center')

    def _plot_box_with_label(self, time_start, time_end,
                             n_qubit_start, n_qubit_end, metadata):
        """
        Parameters
        ----------
        time_start : float
        time_end : float
        n_qubit_start : int
        n_qubit_end : int
        metadata : dict
        """
        box_y = n_qubit_start - 0.5 * _golden_mean
        box_dy = n_qubit_end - n_qubit_start + _golden_mean
        box_x = time_start + 0.5*self.gate_offset
        box_dx = time_end - time_start - self.gate_offset

        label = metadata.pop('label')
        rect = Rectangle((box_x, box_y), box_dx, box_dy,
                         **self._get_box_kwargs(metadata))
        self.ax.add_patch(rect)
        if label is not None:
            self.ax.text(box_x + 0.5 * box_dx, box_y + 0.5 * box_dy,
                         label, ha='center', va='center',
                         zorder=self.zorders['text'])

    def _plot_vline(self, time, n_qubit_start, n_qubit_end, metadata):
        """

        Parameters
        ----------
        time : float
        n_qubit_start : int
        n_qubit_end : int
        metadata : dict
        """
        self.ax.plot((time, time), (n_qubit_start, n_qubit_end),
                     **self._get_line_kwargs(metadata))

    def _plot_qubit_line(self, qubit):
        n = self._qubit_number(qubit)
        self.ax.plot((self.circuit.time_start, self.circuit.time_end),
                     (n, n), color='k')

    def _qubit_number(self, qubit):
        return self.qubits.index(qubit)

    def _qubit_range(self, qubits):
        indices = [self.qubits.index(qubit) for qubit in qubits]
        return min(indices), max(indices)

    def _get_marker_kwargs(self, item):
        if not isinstance(item, dict):
            raise RuntimeError('item must be dict')
        item['color'] = item.pop('color', 'k')
        item['marker'] = item.pop('label', 'o')
        item['zorder'] = item.get('zorder', self.zorders['marker'])
        return item

    def _get_box_kwargs(self, item):
        if not isinstance(item, dict):
            raise RuntimeError('item must be dict')
        item['facecolor'] = item.get('facecolor', 'white')
        item['edgecolor'] = item.get('edgecolor', 'black')
        item['zorder'] = item.get('zorder', self.zorders['box'])
        return item

    def _get_line_kwargs(self, item):
        if not isinstance(item, dict):
            raise RuntimeError('item must be dict')
        item.pop('style', None)
        item['color'] = item.get('color', 'k')
        item['zorder'] = item.get('zorder', self.zorders['line'])
        return item
