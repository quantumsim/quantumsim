import numpy as np

from .. import bases
from ..operations.operation import ParametrizedOperation
from ..operations.qubits import (
    cnot, cphase, swap, hadamard, rotate_x, rotate_y, rotate_z)
from .model import Model
from ..setups import Setup

_basis = bases.general(2),
_basis_classical = bases.general(2).subbasis([0, 1]),


def get_ideal_setup(qubits):
    """
    Initializes an ideal setup file for a given list of qubit.

    Parameters
    ----------
    qubits : list or tuple
        The list of the qubits involved in the circuit.

    Returns
    -------
    quantumsim.Setup
        The setup file containing the ideal gate and qubit parameters.
    """
    _gate_setup = [{
        'time_1qubit': 0,
        'time_2qubit': 0,
        'time_measure': 0}]
    _qubit_setup = [{'qubit': q} for q in qubits]
    return Setup({
        'version': '1',
        'name': 'Ideal Setup',
        'setup': _gate_setup + _qubit_setup})


class IdealModel(Model):
    """
    A model for an ideal error model, where gates are instantaneous and perfect, while the qubits experiences no noise.
    """
    _ptm_project = [rotate_x(0).set_bases(
        (bases.general(2).subbasis([i]),), (bases.general(2).subbasis([i]),)
    ) for i in range(2)]

    dim = 2

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_x(self, qubit):
        """Rotation around the X-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_x(np.deg2rad(angle)),
            _basis).at(qubit),
        )

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_y(self, qubit):
        """Rotation around the Y-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_y(np.deg2rad(angle)),
            _basis).at(qubit),
        )

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_z(self, qubit):
        """Rotation around the Z-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_z(np.deg2rad(angle)),
            _basis).at(qubit),
        )

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$H$'})
    def hadamard(self, qubit):
        """A Hadamard gate.
        """
        return (hadamard().at(qubit),)

    @Model.gate(duration='time_2qubit', plot_metadata={
        'style': 'line',
        'markers': [
            {'style': 'marker', 'label': 'o'},
            {'style': 'marker', 'label': 'o'}
        ]
    })
    def cphase(self, control_qubit, target_qubit):
        """Conditional phase rotation of the target qubit by a given angle, depending on the state of the control qubit. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: cphase(np.deg2rad(angle)),
            _basis * 2).at(control_qubit, target_qubit),
        )

    @Model.gate(duration='time_2qubit', plot_metadata={
        'style': 'line',
        'markers': [
            {'style': 'marker', 'label': 'o'},
            {'style': 'marker', 'label': r'$\oplus$'}
        ]
    })
    def cnot(self, control_qubit, target_qubit):
        """Conditional NOT on the target qubit depending on the state of the control qubit. Parameters: `angle` (degrees).
        """
        return (cnot().at(control_qubit, target_qubit),)

    @Model.gate(duration='time_2qubit', plot_metadata={
        'style': 'line',
        'markers': [
            {'style': 'marker', 'label': 'x'},
            {'style': 'marker', 'label': 'x'}
        ]
    })
    def swap(self, control_qubit, target_qubit):
        """A SWAP gate.
        """
        return (swap().at(control_qubit, target_qubit),)

    @Model.gate(duration='time_measure', plot_metadata={
        'style': 'box', 'label': r'$\circ\!\!\!\!\!\!\!\nearrow$'})
    def measure(self, qubit):
        """A measurement gate.
        """
        def project(result):
            if result in (0, 1):
                return self._ptm_project[result]
            else:
                raise ValueError(
                    'Unknown measurement result: {}'.format(result))

        return (
            ParametrizedOperation(project, _basis_classical).at(qubit),
        )
