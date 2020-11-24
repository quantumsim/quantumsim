import numpy as np

from .. import bases
from ..operations.operation import ParametrizedOperation
from ..operations import qubits as qubit_ops
from ..setups import Setup
from .model import Model

_BASIS = (bases.general(2),)
_BASIS_CLASSICAL = (bases.general(2).subbasis([0, 1]),)


def _born_projection(qubits, state, rng):
    if len(qubits) != 1:
        raise ValueError("Measure should only act on a single qubit")
    ind = state.qubits.index(qubits[0])
    meas_probs = state.pauli_vector.meas_prob(ind)
    meas_probs /= np.sum(meas_probs)
    meas_probs[meas_probs < 0] = 0
    result = rng.choice(len(meas_probs), p=meas_probs)
    return result


class IdealModel(Model):
    """
    A model for an ideal error model, where gates are
    instantaneous and perfect, while the qubits experiences no noise.
    """

    def __init__(self):
        setup = Setup(
            {
                "version": "1",
                "name": "Ideal Setup",
                "setup": [{"time_1qubit": 0, "time_2qubit": 0, "time_measure": 0}],
            }
        )
        super().__init__(setup=setup)

    _ptm_project = [qubit_ops.rotate_x(0).set_bases(
        (bases.general(2).subbasis([i]),), (bases.general(2).subbasis([i]),))
        for i in range(2)]

    dim = 2

    @Model.gate(
        duration="time_1qubit", plot_metadata={"style": "box", "label": "$X({theta})$"}
    )
    def rotate_x(self, qubit):
        """Rotation around the X-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (
            ParametrizedOperation(lambda angle: qubit_ops.rotate_x(np.deg2rad(angle)), _BASIS).at(
                qubit
            ),
        )

    @Model.gate(
        duration="time_1qubit", plot_metadata={"style": "box", "label": "$X({theta})$"}
    )
    def rotate_y(self, qubit):
        """Rotation around the Y-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (
            ParametrizedOperation(lambda angle: qubit_ops.rotate_y(np.deg2rad(angle)), _BASIS).at(
                qubit
            ),
        )

    @Model.gate(
        duration="time_1qubit", plot_metadata={"style": "box", "label": "$X({theta})$"}
    )
    def rotate_z(self, qubit):
        """Rotation around the Z-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (
            ParametrizedOperation(lambda angle: qubit_ops.rotate_z(np.deg2rad(angle)), _BASIS).at(
                qubit
            ),
        )

    @Model.gate(duration="time_1qubit", plot_metadata={"style": "box", "label": "$H$"})
    def hadamard(self, qubit):
        """A Hadamard gate.
        """
        return (hadamard().at(qubit),)

    @Model.gate(
        duration="time_2qubit",
        plot_metadata={
            "style": "line",
            "markers": [
                {"style": "marker", "label": "o"},
                {"style": "marker", "label": "o"},
            ],
        },
    )
    def cphase(self, control_qubit, target_qubit):
        """Conditional phase rotation of the target
        qubit by a given angle, depending on the state of the control qubit.
        Parameters: `angle` (degrees).
        """
        return (
            ParametrizedOperation(
                lambda angle: qubit_ops.cphase(np.deg2rad(angle)), _BASIS * 2
            ).at(control_qubit, target_qubit),
        )

    @Model.gate(
        duration="time_2qubit",
        plot_metadata={
            "style": "line",
            "markers": [
                {"style": "marker", "label": "o"},
                {"style": "marker", "label": r"$\oplus$"},
            ],
        },
    )
    def cnot(self, control_qubit, target_qubit):
        """Conditional NOT on the target qubit depending on the state of the control qubit. Parameters: `angle` (degrees).
        """
        return (qubit_ops.cnot().at(control_qubit, target_qubit),)

    @Model.gate(
        duration="time_2qubit",
        plot_metadata={
            "style": "line",
            "markers": [
                {"style": "marker", "label": "x"},
                {"style": "marker", "label": "x"},
            ],
        },
    )
    def swap(self, control_qubit, target_qubit):
        """A SWAP gate.
        """
        return (qubit_ops.swap().at(control_qubit, target_qubit),)

    @Model.gate(
        duration="time_measure",
        plot_metadata={"style": "box",
                       "label": r"$\circ\!\!\!\!\!\!\!\nearrow$"},
        param_funcs={"result": _born_projection},
    )
    def measure(self, qubit):
        """A measurement gate.
        """

        def project(result):
            if result in (0, 1):
                return self._ptm_project[result]
            raise ValueError("Unknown measurement result: {}".format(result))

        return (ParametrizedOperation(project, _BASIS_CLASSICAL).at(qubit),)
