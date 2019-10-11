import pytest
import sympy

from numpy import pi
from pytest import approx
import quantumsim.operations.qubits as lib
from quantumsim.circuits import TimeAgnosticGate, TimeAwareGate, \
    allow_param_repeat
from quantumsim.circuits.circuit import ParametrizedOperation
from quantumsim import bases, Operation

bases1q = (bases.general(2),)
bases2q = bases1q * 2

def c_op(circuit):
    return circuit.finalize().operation

# noinspection PyTypeChecker
@pytest.mark.parametrize('cls', [TimeAgnosticGate, TimeAwareGate])
class TestCircuitsCommon:
    def test_gate_create_no_params(self, cls):
        angle = 1.0758
        dim = 2

        op_1q = lib.rotate_x(angle)
        op_2q = lib.cphase(angle)

        gate = cls('qubit', dim, op_1q)
        assert c_op(gate) == op_1q
        assert gate.qubits == ('qubit',)
        assert len(gate.free_parameters) == 0

        gate = cls('Q0', dim, op_1q)
        assert c_op(gate) == op_1q
        assert gate.qubits == ('Q0',)
        assert len(gate.free_parameters) == 0

        gate = cls(('Q1',), dim, op_1q)
        assert c_op(gate) == op_1q
        assert gate.qubits == ('Q1',)
        assert len(gate.free_parameters) == 0

        gate = cls(('D', 'A'), dim, op_2q)
        assert c_op(gate) == op_2q
        assert gate.qubits == ('D', 'A')
        assert len(gate.free_parameters) == 0

    def test_gate_params_call(self, cls):
        dim = 2
        angle_cphase, angle_rotate = sympy.symbols('angle_cphase angle_rotate')

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

        gate = cls(('D', 'A'), dim,
                   ParametrizedOperation(cnot_like, basis, basis))
        assert gate.free_parameters == {angle_cphase, angle_rotate}

        gate1 = gate(angle_cphase=angle1_cphase, angle_rotate=angle1_rotate)
        assert gate.free_parameters == {angle_cphase, angle_rotate}
        assert gate1.free_parameters == set()

        gate2 = gate(angle_cphase=angle2_cphase)
        assert gate.free_parameters == {angle_cphase, angle_rotate}
        assert gate2.free_parameters == {angle_rotate}

        assert gate1.finalize().operation.ptm(basis, basis) == approx(gate(
            angle_cphase=angle1_cphase, angle_rotate=angle1_rotate
        ).finalize().operation.ptm(basis, basis))
        assert (gate2.finalize()(angle_rotate=angle2_rotate).operation
                .ptm(basis, basis)) == approx(
            gate.finalize()(
                angle_cphase=angle2_cphase, angle_rotate=angle2_rotate
            ).operation.ptm(basis, basis))

    def test_circuits_add(self, cls):
        dim = 2
        orplus = lib.rotate_y(0.5 * pi)
        ocphase = lib.cphase(pi)
        orminus = lib.rotate_y(-0.5 * pi)
        grplus = cls('Q0', dim, orplus)
        gcphase = cls(('Q0', 'Q1'), dim, ocphase)
        grminus = cls('Q0', dim, orminus)
        basis = (bases.general(2),) * 2

        circuit = grplus + gcphase
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 2
        assert c_op(circuit).ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1)
            ).ptm(basis, basis))

        circuit = circuit + grminus
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert c_op(circuit).ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        circuit = grplus + (gcphase + grminus)
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert c_op(circuit).ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        grplus = cls('Q1', dim, orplus)
        grminus = cls('Q1', dim, orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q1', 'Q0')
        assert len(circuit.gates) == 3
        assert c_op(circuit).ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        basis = (basis[0],) * 3

        grplus = cls('Q2', dim, orplus)
        grminus = cls('Q0', dim, orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q2', 'Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert c_op(circuit).ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(1, 2), orminus.at(1)
            ).ptm(basis, basis))

        grplus = cls('Q0', dim, orplus)
        grminus = cls('Q2', dim, orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q0', 'Q1', 'Q2')
        assert len(circuit.gates) == 3
        assert c_op(circuit).ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(2)
            ).ptm(basis, basis))

    def test_circuits_params(self, cls):
        dim = 2
        basis = (bases.general(dim),) * 2
        orotate = lib.rotate_y
        ocphase = lib.cphase
        grotate = cls('Q0', dim, ParametrizedOperation(orotate, basis[:1]))
        gcphase = cls(('Q0', 'Q1'), dim, ParametrizedOperation(ocphase, basis))

        with pytest.raises(RuntimeError,
                           match=r".*free parameters.*\n"
                                 r".*angle.*\n"
                                 r".*allow_param_repeat.*"):
            _ = gcphase + grotate

        with allow_param_repeat():
            circuit = grotate + gcphase + grotate

        assert circuit.free_parameters == {sympy.symbols('angle')}
        assert len(circuit.gates) == 3
        angle = 0.736
        assert c_op(circuit(angle=angle)).ptm(basis, basis) == \
               approx(Operation.from_sequence(
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
        assert circuit.finalize().operation.ptm(basis, basis) == \
               approx(ptm_ref)

        circuit = grotate(angle='angle1') + gcphase(angle='angle2') + \
                  grotate(angle='angle3')
        assert c_op(circuit(
            angle1=angle1, angle2=angle2, angle3=angle3
        )).ptm(basis, basis) == approx(ptm_ref)


class TestCircuitsTimeAware:
    def test_gate_create(self):
        dim = 2
        gate = TimeAwareGate('Q0', dim, lib.rotate_y(0.5 * pi), 20.)
        assert gate.time_start == 0
        assert gate.time_end == 20.
        assert gate.duration == 20.

        gate = TimeAwareGate(('Q0', 'Q1'), dim, lib.cphase(0.5 * pi), 40., 125.)
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
        dim = 2
        orplus = lib.rotate_y(0.5 * pi)
        ocphase = lib.cphase(pi)
        orminus = lib.rotate_y(-0.5 * pi)
        gate_q0 = TimeAwareGate('Q0', dim, orplus, 20.)
        gate_2q = TimeAwareGate(('Q0', 'Q1'), dim, ocphase, 40.)
        gate_q1 = TimeAwareGate('Q1', dim, orminus, 30.)

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
        dim = 2
        big_gate = TimeAwareGate('A', dim, lib.rotate_x(0.), 600.)
        rotx1 = TimeAwareGate('D0', dim, lib.rotate_x(pi), 20., 290.)
        rotx2 = TimeAwareGate('A', dim, lib.rotate_x(pi), 20., )

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
