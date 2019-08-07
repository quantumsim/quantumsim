import pytest

from numpy import pi
from pytest import approx
import quantumsim.operations.qubits as lib
from quantumsim.circuits import TimeAgnosticGate, TimeAwareGate, \
    allow_param_repeat
from quantumsim import bases, Operation


# noinspection PyTypeChecker
@pytest.mark.parametrize('cls', [TimeAgnosticGate, TimeAwareGate])
class TestCircuitsCommon:
    def test_gate_create_no_params(self, cls):
        angle = 1.0758

        op_1q = lib.rotate_x(angle)
        op_2q = lib.cphase(angle)

        gate = cls('qubit', op_1q)
        assert gate.operation == op_1q
        assert gate.qubits == ('qubit',)
        assert len(gate.params) == 0

        gate = cls('Q0', op_1q)
        assert gate.operation == op_1q
        assert gate.qubits == ('Q0',)
        assert len(gate.params) == 0

        gate = cls(('Q1',), op_1q)
        assert gate.operation == op_1q
        assert gate.qubits == ('Q1',)
        assert len(gate.params) == 0

        gate = cls(('D', 'A'), op_2q)
        assert gate.operation == op_2q
        assert gate.qubits == ('D', 'A')
        assert len(gate.params) == 0

    def test_gate_params_call(self, cls):
        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib.rotate_y(angle_rotate).at(1),
                lib.cphase(angle_cphase).at(0, 1),
                lib.rotate_y(-angle_rotate).at(1))

        basis = (bases.general(2),) * 2
        angle1_cphase = 1.2 * pi
        angle1_rotate = 0.6 * pi
        angle2_cphase = 0.8 * pi
        angle2_rotate = 0.3 * pi

        gate = cls(('D', 'A'), ParametrizedOperation(cnot_like, basis, basis))
        gate1 = gate(angle_cphase=angle1_cphase, angle_rotate=angle1_rotate)
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate1.params == set()

        gate2 = gate(angle_cphase=angle2_cphase)
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate2.params == {'angle_rotate'}

        assert gate1.operation.ptm(basis, basis) == approx(gate(
            angle_cphase=angle1_cphase, angle_rotate=angle1_rotate
        ).operation.ptm(basis, basis))
        assert (gate2(angle_rotate=angle2_rotate).operation
                .ptm(basis, basis)) == approx(
            gate(angle_cphase=angle2_cphase, angle_rotate=angle2_rotate)
            .operation.ptm(basis, basis))

    def test_circuits_add(self, cls):
        orplus = lib.rotate_y(0.5 * pi)
        ocphase = lib.cphase(pi)
        orminus = lib.rotate_y(-0.5 * pi)
        grplus = cls('Q0', orplus)
        gcphase = cls(('Q0', 'Q1'), ocphase)
        grminus = cls('Q0', orminus)
        basis = (bases.general(2),) * 2

        circuit = grplus + gcphase
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 2
        assert circuit.operation.ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1)
            ).ptm(basis, basis))

        circuit = circuit + grminus
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert circuit.operation.ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        circuit = grplus + (gcphase + grminus)
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert circuit.operation.ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        grplus = cls('Q1', orplus)
        grminus = cls('Q1', orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q1', 'Q0')
        assert len(circuit.gates) == 3
        assert circuit.operation.ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        basis = (basis[0],) * 3

        grplus = cls('Q2', orplus)
        grminus = cls('Q0', orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q2', 'Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert circuit.operation.ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(1, 2), orminus.at(1)
            ).ptm(basis, basis))

        grplus = cls('Q0', orplus)
        grminus = cls('Q2', orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q0', 'Q1', 'Q2')
        assert len(circuit.gates) == 3
        assert circuit.operation.ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(2)
            ).ptm(basis, basis))

    def test_circuits_params(self, cls):
        basis = (bases.general(2),) * 2
        orotate = lib.rotate_y
        ocphase = lib.cphase
        grotate = cls('Q0', ParametrizedOperation(orotate, basis[:1]))
        gcphase = cls(('Q0', 'Q1'), ParametrizedOperation(ocphase, basis))

        with pytest.raises(RuntimeError,
                           match=r".*free parameters.*\n"
                                 r".*angle.*\n"
                                 r".*allow_param_repeat.*"):
            _ = gcphase + grotate

        with allow_param_repeat():
            circuit = grotate + gcphase + grotate

        assert circuit.params == {'angle'}
        assert len(circuit.gates) == 3
        angle = 0.736
        assert circuit(angle=angle).operation.ptm(basis, basis) == approx(
            Operation.from_sequence(
                orotate(angle).at(0), ocphase(angle).at(0, 1),
                orotate(angle).at(0)
            ).ptm(basis, basis))

        angle1 = 0.4 * pi
        angle2 = 1.01 * pi
        angle3 = -0.6 * pi
        ptm_ref = Operation.from_sequence(
            orotate(angle1).at(0), ocphase(angle2).at(0, 1),
            orotate(angle3).at(0)
        ).ptm(basis, basis)

        circuit = grotate(angle=angle1) + gcphase(angle=angle2) + \
                  grotate(angle=angle3)
        assert circuit.operation.ptm(basis, basis) == approx(ptm_ref)

        circuit = grotate(angle='angle1') + gcphase(angle='angle2') + \
                  grotate(angle='angle3')
        assert circuit(
            angle1=angle1, angle2=angle2, angle3=angle3
        ).operation.ptm(basis, basis) == approx(ptm_ref)


class TestCircuitsTimeAware:
    def test_gate_create(self):
        gate = TimeAwareGate('Q0', lib.rotate_y(0.5 * pi), 20.)
        assert gate.time_start == 0
        assert gate.time_end == 20.
        assert gate.duration == 20.

        gate = TimeAwareGate(('Q0', 'Q1'), lib.cphase(0.5 * pi), 40., 125.)
        assert gate.time_start == 125.
        assert gate.time_end == approx(165.)
        assert gate.duration == 40.

        gate.time_end = 90.
        assert gate.time_start == approx(50.)
        assert gate.time_end == 90.
        assert gate.duration == 40.

        gate1 = gate.shift(time_start=0.)
        assert gate.time_start == approx(50.)
        assert gate.time_end == 90.
        assert gate.duration == 40.
        assert gate1.time_start == 0.
        assert gate1.time_end == 40.
        assert gate1.duration == 40.

        gate1 = gate.shift(time_end=123.)
        assert gate.time_start == approx(50.)
        assert gate.time_end == 90.
        assert gate.duration == 40.
        assert gate1.time_start == approx(83.)
        assert gate1.time_end == 123.
        assert gate1.duration == 40.

    def test_time_aware_add_gate_no_init_time(self):
        orplus = lib.rotate_y(0.5 * pi)
        ocphase = lib.cphase(pi)
        orminus = lib.rotate_y(-0.5 * pi)
        gate_q0 = TimeAwareGate('Q0', orplus, 20.)
        gate_2q = TimeAwareGate(('Q0', 'Q1'), ocphase, 40.)
        gate_q1 = TimeAwareGate('Q1', orminus, 30.)

        circuit = gate_q0 + gate_2q
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(60.)
        assert circuit.duration == approx(60.)
        assert [gate.time_start for gate in circuit.gates] == approx([0., 20.])

        circuit = circuit + gate_q1
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(90.)
        assert circuit.duration == approx(90.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 20., 60.])

        circuit = circuit + gate_q0
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(90.)
        assert circuit.duration == approx(90.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 20., 60., 60.])

        circuit = gate_q1 + circuit
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(100.)
        assert circuit.duration == approx(100.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 10., 30., 70., 70.])

        circuit = circuit + (gate_q0 + gate_q1)
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(130.)
        assert circuit.duration == approx(130.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 10., 30., 70., 70., 100., 100.])

    def test_time_aware_add_gate_and_delays(self):
        big_gate = TimeAwareGate('A', lib.rotate_x(0.), 600.)
        rotx1 = TimeAwareGate('D0', lib.rotate_x(pi), 20., 290.)
        rotx2 = TimeAwareGate('A', lib.rotate_x(pi), 20., )

        circuit1 = rotx2.shift(time_start=20.) + \
                   rotx2.shift(time_end=120.)
        assert circuit1.time_start == 20.
        assert circuit1.time_end == approx(160.)
        assert circuit1.duration == approx(140.)
        assert [gate.time_start for gate in circuit1.gates] == \
               approx([20., 140.])

        circuit2 = big_gate + rotx1
        assert circuit2.time_start == 0.
        assert circuit2.time_end == approx(600.)
        assert circuit2.duration == approx(600.)
        assert [gate.time_start for gate in circuit2.gates] == \
               approx([0., 290.])

        circuit = circuit1 + circuit2
        assert circuit.time_start == 20.
        assert circuit.time_end == approx(760.)
        assert circuit.duration == approx(740.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([20., 140., 160, 450.])
