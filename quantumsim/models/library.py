import numpy as np

from .. import bases
from ..operations.operation import ParametrizedOperation
from ..operations.qubits import (
    cnot, cphase, swap, hadamard, rotate_x, rotate_y, rotate_z, amp_phase_damping)
from .model import Model
from ..setups import Setup

_BASIS = bases.general(2),
_BASIS_CLASSICAL = bases.general(2).subbasis([0, 1]),


def _born_projection(state, rng, *, atol=1e-08, **kwargs):
    meas_probs = state.pauli_vector.diagonal()
    meas_probs[np.abs(meas_probs) < atol] = 0
    meas_probs /= np.sum(meas_probs)
    result = rng.choice(len(meas_probs), p=meas_probs)
    return result


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
    A model for an ideal error model, where gates are
    instantaneous and perfect, while the qubits experiences no noise.
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
            _BASIS).at(qubit),
        )

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_y(self, qubit):
        """Rotation around the Y-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_y(np.deg2rad(angle)),
            _BASIS).at(qubit),
        )

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_z(self, qubit):
        """Rotation around the Z-axis by a given angle. Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: rotate_z(np.deg2rad(angle)),
            _BASIS).at(qubit),
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
        """Conditional phase rotation of the target
        qubit by a given angle, depending on the state of the control qubit.
        Parameters: `angle` (degrees).
        """
        return (ParametrizedOperation(
            lambda angle: cphase(np.deg2rad(angle)),
            _BASIS * 2).at(control_qubit, target_qubit),
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

    @Model.gate(
        duration='time_measure',
        plot_metadata={
            'style': 'box',
            'label': r'$\circ\!\!\!\!\!\!\!\nearrow$'},
        param_funcs={'result': _born_projection})
    def measure(self, qubit):
        """A measurement gate.
        """
        def project(result):
            if result in (0, 1):
                return self._ptm_project[result]
            raise ValueError(
                'Unknown measurement result: {}'.format(result))

        return (
            ParametrizedOperation(project, _BASIS_CLASSICAL).at(qubit),
        )


def get_transmon_setup(qubits):
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
        'time_1qubit': 20,
        'time_2qubit': 40,
        'time_measure': 500}]
    _qubit_setup = [{
        'qubit': q,
        'T1': 30000,
        'T2': 30000,
    } for q in qubits]
    return Setup({
        'version': '1',
        'name': 'Transmon Setup',
        'setup': _gate_setup + _qubit_setup})


class TransmonModel(Model):
    """
    A model for transmon qubits.
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
        t1 = self.p('T1', qubit)
        t2 = self.p('T2', qubit)
        half_duration = 0.5*self.p('time_1qubit', qubit)
        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(
                lambda angle: rotate_x(np.deg2rad(angle)),
                _BASIS).at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_y(self, qubit):
        """Rotation around the Y-axis by a given angle. Parameters: `angle` (degrees).
        """
        t1 = self.p('T1', qubit)
        t2 = self.p('T2', qubit)
        half_duration = 0.5 * self.p('time_1qubit', qubit)

        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(
                lambda angle: rotate_y(np.deg2rad(angle)),
                _BASIS).at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    @Model.gate(duration='time_1qubit', plot_metadata={
        'style': 'box', 'label': '$X({theta})$'})
    def rotate_z(self, qubit):
        """Rotation around the Z-axis by a given angle. Parameters: `angle` (degrees).
        """
        t1 = self.p('T1', qubit)
        t2 = self.p('T2', qubit)
        half_duration = 0.5*self.p('time_1qubit', qubit)

        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(
                lambda angle: rotate_z(np.deg2rad(angle)),
                _BASIS).at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

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
        ctrl_q_t1 = self.p('T1', control_qubit)
        ctrl_q_t2 = self.p('T2', control_qubit)

        target_q_t1 = self.p('T1', target_qubit)
        target_q_t2 = self.p('T2', target_qubit)

        half_duration = 0.5*self.p('time_2qubit', control_qubit, target_qubit)
        return (
            self._idle(control_qubit, half_duration, ctrl_q_t1, ctrl_q_t2),
            self._idle(target_qubit, half_duration, target_q_t1, target_q_t2),
            ParametrizedOperation(
                lambda angle: cphase(np.deg2rad(angle)),
                _BASIS * 2).at(control_qubit, target_qubit),
            self._idle(control_qubit, half_duration, ctrl_q_t1, ctrl_q_t2),
            self._idle(target_qubit, half_duration, target_q_t1, target_q_t2),
        )

    @Model.gate(duration='time_measure', plot_metadata={
        'style': 'box', 'label': r'$\circ\!\!\!\!\!\!\!\nearrow$'})
    def measure(self, qubit):
        """A measurement gate.
        """

        t1 = self.p('T1', qubit)
        t2 = self.p('T2', qubit)
        half_duration = 0.5 * self.p('time_measure', qubit)

        def project(result):
            if result in (0, 1):
                return self._ptm_project[result]
            else:
                raise ValueError(
                    'Unknown measurement result: {}'.format(result))

        return (
            self._idle(qubit, half_duration, t1, t2),
            ParametrizedOperation(project, _BASIS_CLASSICAL).at(qubit),
            self._idle(qubit, half_duration, t1, t2),
        )

    def _idle(self, qubit, duration, t1, t2):
        """
        An idling gate.
        """
        if np.isfinite(t1) and np.isfinite(t2):
            if t1 <= 0:
                raise ValueError('T1 must be greater than 0')
            damp_rate = 1 - np.exp(-duration/t1)

            if t2 <= 0:
                raise ValueError('T2 must be greater than 0')

            deph_rate = 1. / t2 - 0.5 / t1
            if deph_rate < 0:
                raise ValueError('t2 must be less than 2*T1')

            return amp_phase_damping(damp_rate, deph_rate).at(qubit)

        return amp_phase_damping(0, 0).at(qubit)
