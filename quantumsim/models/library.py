import numpy as np

from .. import bases
# from ..operations.operation import ParametrizedOperation
from ..circuits import Gate
from ..models.perfect.qubits import (
    cnot,
    cphase,
    hadamard,
    rotate_x,
    rotate_y,
    rotate_z,
    swap,
)
from ..setups import Setup
from .model import Model

_BASIS = (bases.general(2),)
_BASIS_CLASSICAL = (bases.general(2).subbasis([0, 1]),)


def _born_projection(inds, state, rng, *, atol=1e-08):
    if len(inds) != 1:
        raise ValueError("Measure should only act on a single qubit")
    meas_probs = np.array(state.pauli_vector.meas_prob(*inds))
    meas_probs[np.abs(meas_probs) < atol] = 0
    meas_probs /= np.sum(meas_probs)
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
                "setup": [],
            }
        )
        super().__init__(setup=setup)

    _ptm_project = [rotate_x(0).set_bases(
        (bases.general(2).subbasis([i]),), (bases.general(2).subbasis([i]),))
        for i in range(2)]

    dim = 2

    @Model.gate(
        plot_metadata={"style": "box", "label": "$X({theta})$"},
    )
    def rotate_x(self, qubit):
        """Rotation around the X-axis by a given angle. Parameters: `angle` (degrees).
        """
        return rotate_x(qubit)

    @Model.gate(
        plot_metadata={"style": "box", "label": "$X({theta})$"}
    )
    def rotate_y(self, qubit):
        """Rotation around the Y-axis by a given angle. Parameters: `angle` (degrees).
        """
        return rotate_y(qubit)

    @Model.gate(
        plot_metadata={"style": "box", "label": "$X({theta})$"}
    )
    def rotate_z(self, qubit):
        """Rotation around the Z-axis by a given angle. Parameters: `angle` (degrees).
        """
        return rotate_z(qubit)

    @Model.gate(
        plot_metadata={"style": "box", "label": "$H$"},
    )
    def hadamard(self, qubit):
        """A Hadamard gate.
        """
        return hadamard(qubit)

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
    def cphase(self, qubit1, qubit2):
        """Conditional phase rotation of the target
        qubit by a given angle, depending on the state of the control qubit.
        Parameters: `angle` (degrees).
        """
        return cphase(qubit1, qubit2)

    @Model.gate(
        plot_metadata={
            "style": "line",
            "markers": [
                {"style": "marker", "label": "o"},
                {"style": "marker", "label": r"$\oplus$"},
            ],
        },
    )
    def cnot(self, control_qubit, target_qubit):
        """Conditional NOT on the target qubit depending on the state of the control
        qubit. Parameters: `angle` (degrees).
        """
        return cnot(control_qubit, target_qubit)

    @Model.gate(
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
        return swap(control_qubit, target_qubit)

    @Model.gate(
        plot_metadata={"style": "box",
                       "label": r"$\circ\!\!\!\!\!\!\!\nearrow$"},
        param_funcs={"result": _born_projection},
    )
    def measure(self, qubit):
        """A measurement gate.
        """

        def project(result):
            if result in (0, 1):
                basis_element = (_BASIS[0].subbasis([result]),)
                return np.array([[1.]]), basis_element, basis_element
            raise ValueError("Unknown measurement result: {}".format(result))

        return Gate(qubit,
                    self.dim,
                    project,
                    duration=0,
                    plot_metadata={"style": "box",
                                   "label": r"$\circ\!\!\!\!\!\!\!\nearrow$"},
                    repr_='measure',
                    )
