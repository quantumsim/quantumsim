from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Rectangle
from matplotlib import colorbar as _colorbar
from sympy import latex

_golden_mean = (np.sqrt(5) - 1.0) / 2.0


def plot(circuit, *, ax=None, qubit_order=None, gate_offset=2.0):
    plotter = MatplotlibPlotter(circuit, ax, qubit_order, gate_offset)
    return plotter.plot()


class MatplotlibPlotter:
    zorders = {
        "line": 1,
        "marker": 1,
        "box": 10,
        "text": 20,
    }

    def __init__(self, circuit, ax, qubit_order, gate_offset, figsize=None):
        self.circuit = circuit
        self.gate_offset = 0.5 * gate_offset

        if ax is not None:
            self.fig = None
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(
                figsize=figsize or (7, 0.5 * len(self.circuit.qubits))
            )
            self.ax.set_ylim(-1, len(self.circuit.qubits))
            self.ax.set_yticks([])
            t_min = self.circuit.time_start
            t_max = self.circuit.time_end

            if t_max - t_min < 0.1:
                t_min -= 0.05
                t_max += 0.05
            buffer = (t_max - t_min) * 0.05
            self.ax.set_xlim(t_min - 2.5 * buffer, t_max + 1.5 * buffer)

        if callable(qubit_order):
            self.qubits = sorted(circuit.qubits, key=qubit_order)
        elif hasattr(qubit_order, "__iter__"):
            self.qubits = tuple(qubit_order)
        elif qubit_order is None:
            self.qubits = sorted(circuit.qubits)
        else:
            raise ValueError("Qubit order must be a list, callable or None")

    def plot(self):
        for qubit in self.circuit.qubits:
            self._plot_qubit_line(qubit)
            self._annotate_qubit(qubit)
        for gate in self.circuit.gates:
            self._plot_gate(gate)
        return self.fig

    def _plot_single_qubit_marker(self, qubit, time_start, duration, marker_dict):
        if marker_dict is None:
            return
        if not isinstance(marker_dict, dict):
            raise RuntimeError("marker_dict must be dict")
        style = marker_dict.pop("style", "marker")
        n = self._qubit_number(qubit)
        if style == "marker":
            time = time_start + 0.5 * duration
            marker_kwargs = self._get_marker_kwargs(marker_dict)
            self.ax.scatter((time,), (self._qubit_number(qubit),), **marker_kwargs)
        elif style == "box":
            label = marker_dict.pop("label", "")
            return self._plot_box_with_label(
                time_start,
                time_start + duration,
                n,
                n,
                label,
                **self._get_box_kwargs(marker_dict)
            )
        else:
            raise RuntimeError("Unknown marker style: {}".format(style))

    def _plot_gate(self, gate):
        """

        Parameters
        ----------
        gate : quantumsim.circuits.PlotUnitMixin

        Returns
        -------

        """
        metadata = deepcopy(gate.plot_metadata)
        # By default we will plot a box
        style = metadata.pop("style", "box")

        if style == "box":
            # TODO: formatting with params (it is tricky)
            # params = gate.params_set()
            label = metadata.pop("label", r"$\mathcal{{G}}$")  # .format(**params)
            params = {key: latex(val) for key, val in gate.params.items()}
            label = label.format(**params)
            return self._plot_box_with_label(
                gate.time_start,
                gate.time_end,
                *self._qubit_range(gate.qubits),
                label,
                **self._get_box_kwargs(metadata)
            )
        elif style == "line":
            time = gate.time_start + 0.5 * gate.duration
            markers = metadata.pop("markers")
            self._plot_vline(time, *self._qubit_range(gate.qubits), metadata)
            if markers is not None:
                for qubit, marker in zip(gate.qubits, markers):
                    self._plot_single_qubit_marker(
                        qubit, gate.time_start, gate.duration, marker
                    )
        elif style == "marker":
            for qubit in gate.qubits:
                self._plot_single_qubit_marker(
                    qubit, gate.time_start, gate.duration, metadata
                )
        else:
            raise RuntimeError("Unknown gate plotting style: {}".format(style))

    def _annotate_qubit(self, qubit):
        xlim = self.ax.get_xlim()
        xlim = xlim[1] - xlim[0]
        time = self.circuit.time_start - 0.01 * xlim
        self.ax.text(time, self._qubit_number(qubit), qubit, ha="right", va="center")

    def _plot_box_with_label(
        self, time_start, time_end, n_qubit_start, n_qubit_end, label, **kwargs
    ):
        """
        Parameters
        ----------
        time_start : float
        time_end : float
        n_qubit_start : int
        n_qubit_end : int
        label : str
        kwargs
        """
        box_y = n_qubit_start - 0.5 * _golden_mean
        box_dy = n_qubit_end - n_qubit_start + _golden_mean
        box_x = time_start + 0.5 * self.gate_offset
        box_dx = time_end - time_start - self.gate_offset

        rect = Rectangle((box_x, box_y), box_dx, box_dy, **kwargs)
        self.ax.add_patch(rect)
        if label is not None:
            self.ax.text(
                box_x + 0.5 * box_dx,
                box_y + 0.5 * box_dy,
                label,
                ha="center",
                va="center",
                zorder=self.zorders["text"],
            )

    def _plot_vline(self, time, n_qubit_start, n_qubit_end, metadata):
        """

        Parameters
        ----------
        time : float
        n_qubit_start : int
        n_qubit_end : int
        metadata : dict
        """
        self.ax.plot(
            (time, time),
            (n_qubit_start, n_qubit_end),
            **self._get_line_kwargs(metadata)
        )

    def _plot_qubit_line(self, qubit):
        n = self._qubit_number(qubit)
        self.ax.plot(
            (self.circuit.time_start, self.circuit.time_end), (n, n), color="k"
        )

    def _qubit_number(self, qubit):
        return self.qubits.index(qubit)

    def _qubit_range(self, qubits):
        indices = [self.qubits.index(qubit) for qubit in qubits]
        return min(indices), max(indices)

    def _get_marker_kwargs(self, item):
        if not isinstance(item, dict):
            raise RuntimeError("item must be dict")
        item["color"] = item.pop("color", "k")
        item["marker"] = item.pop("label", "o")
        item["zorder"] = item.get("zorder", self.zorders["marker"])
        return item

    def _get_box_kwargs(self, item):
        if not isinstance(item, dict):
            raise RuntimeError("item must be dict")
        item["facecolor"] = item.get("facecolor", "white")
        item["edgecolor"] = item.get("edgecolor", "black")
        item["zorder"] = item.get("zorder", self.zorders["box"])
        return item

    def _get_line_kwargs(self, item):
        if not isinstance(item, dict):
            raise RuntimeError("item must be dict")
        item.pop("style", None)
        item["color"] = item.get("color", "k")
        item["zorder"] = item.get("zorder", self.zorders["line"])
        return item


def circuit_heatmap(
    circuit, bases_in=None, bases_out=None, ax=None, truncate_levels=None, colorbar=True
):
    """
    Parameters
    ----------
    circuit : quantumsim.Circuit
        Operation to display
    bases_in, bases_out : tuple of quantumsim.PauliBasis, optional
    ax : matplotlib.axes.Axes or None
        Axes to plot onto. If None, new figure is created and returned.
    truncate_levels : None or int
        If not None, all the states higher than provided are discarded and a
        identity is added to the state instead, so that total trace is
        preserved. This should emulate behavior of tomography in the presence
        of leakage.
    colorbar: bool

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = None

    dim = circuit.dim_hilbert
    num_qubits = circuit.num_qubits
    num_basis_elements = (dim**2) ** num_qubits

    bases_in = bases_in or circuit.bases_in
    bases_out = bases_out or circuit.bases_out

    ptm = circuit.ptm(bases_in, bases_out).reshape(
        num_basis_elements, num_basis_elements
    )

    if truncate_levels is not None:
        raise NotImplementedError

    def tuple_to_string(tup):
        pauli_element = "".join(str(x) for x in tup)
        return r"$%s$" % pauli_element

    bases_in_labels = (basis.labels for basis in bases_in)
    x_labels = [tuple_to_string(x) for x in product(*bases_in_labels)]

    bases_out_labels = (basis.labels for basis in bases_out)
    y_labels = [tuple_to_string(x) for x in product(*bases_out_labels)]

    img = ax.imshow(ptm, cmap="bwr", aspect="equal", origin="upper")
    img.set_clim(vmin=-1, vmax=1)

    ax.set_xticks(range(num_basis_elements))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Input basis")

    ax.set_yticks(range(num_basis_elements))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Output basis")

    if colorbar:
        cax, _ = _colorbar.make_axes(ax)
        cbar = _colorbar.Colorbar(cax, img)
        cbar.set_ticks((-1, 0, 1))
        cbar.ax.set_ylabel("Amplitude", rotation=270)

    return fig
