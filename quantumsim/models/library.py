import numpy as np

from .. import bases
from ..operations.operation import ParametrizedOperation
from ..operations.qubits import (
    cnot, cphase, hadamard, rotate_x, rotate_y, rotate_z)
from .model import Model

_basis = bases.general(2)
_basis_classical = bases.general(2).subbasis([0, 1]),


class IdealModel(Model):
    _ptm_project = [rotate_x(0).set_bases(
        (bases.general(2).subbasis([i]),), (bases.general(2).subbasis([i]),)
    ) for i in range(2)]

    dim = 2

    def __init__(self):
        setup = None
        super().__init__(setup)

    @Model.gate(duration=0, plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_x(self, qubit):
        """Rotation around the X-axis Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_x(np.deg2rad(angle)),
            _basis).at(qubit),
        )

    @Model.gate(duration=0, plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_y(self, qubit):
        """Rotation around the Y-axis. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_y(np.deg2rad(angle)),
            _basis).at(qubit),
        )

    @Model.gate(duration=0, plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_z(self, qubit):
        """Rotation around the Z-axis. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_z(np.deg2rad(angle)),
            _basis).at(qubit),
        )

    @Model.gate(duration=0, plot_metadata={
        'style': 'box', 'label': '$H$'})
    def hadamard(self, qubit):
        """Rotation around the Z-axis. Parameters: `angle` (degrees).
        """
        return (hadamard().at(qubit),)

    @Model.gate(duration=0, plot_metadata={
        'style': 'line',
        'markers': [
            {'style': 'marker', 'label': 'o'},
            {'style': 'marker', 'label': 'o'}
        ]
    })
    def cphase(self, control_qubit, target_qubit):
        """Rotation around the Z-axis. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: cphase(np.deg2rad(angle)),
            _basis * 2).at(control_qubit, target_qubit),
        )

    @Model.gate(duration=0, plot_metadata={
        'style': 'line',
        'markers': [
            {'style': 'marker', 'label': 'o'},
            {'style': 'marker', 'label': r'$\oplus$'}
        ]
    })
    def cnot(self, control_qubit, target_qubit):
        """Rotation around the Z-axis. Parameters: `angle` (degrees).
        """
        return (cnot().at(control_qubit, target_qubit),)

    @Model.gate(duration=0, plot_metadata={
        'style': 'box', 'label': r'$\circ\!\!\!\!\!\!\!\nearrow$'})
    def measure(self, qubit):
        def project(result):
            if result in (0, 1):
                return self._ptm_project[result]
            else:
                raise ValueError(
                    'Unknown measurement result: {}'.format(result))

        return (
            ParametrizedOperation(project, _basis_classical).at(qubit),
        )
