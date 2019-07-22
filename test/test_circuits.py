import pytest


from numpy import pi
from pytest import approx
import quantumsim.models.qubits as lib
from quantumsim.circuits import TimeAgnosticGate, allow_parameter_collisions
from quantumsim import bases, Operation


# noinspection PyTypeChecker
class TestCircuits:
    def test_gate_create_no_params(self):
        angle = 1.0758

        op_1q = lib.rotate_x(angle)
        op_2q = lib.cphase(angle)

        gate = TimeAgnosticGate('qubit', op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('qubit',)
        assert len(gate.params) == 0

        gate = TimeAgnosticGate('Q0', lambda: op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('Q0',)
        assert len(gate.params) == 0

        gate = TimeAgnosticGate(('Q1',), lambda: op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('Q1',)
        assert len(gate.params) == 0

        gate = TimeAgnosticGate(('D', 'A'), lambda: op_2q)
        assert gate.operation() == op_2q
        assert gate.qubits == ('D', 'A')
        assert len(gate.params) == 0

        with pytest.raises(RuntimeError,
                           match=".*number of qubits does not match .*"):
            TimeAgnosticGate('Q0', op_2q).operation()
        with pytest.raises(RuntimeError,
                           match=".*number of qubits does not match .*"):
            TimeAgnosticGate(('Q0', 'Q1'), lambda: op_1q).operation()
        with pytest.raises(RuntimeError,
                           match="Invalid operation function was provided.*"):
            TimeAgnosticGate('Q0', lambda: [[1., 0.], [0., 1.]]).operation()

        with pytest.raises(ValueError,
                           match=".*can't accept free arguments.*"):
            TimeAgnosticGate('Q0', lambda *args: op_1q)
        with pytest.raises(ValueError,
                           match=".*can't accept free keyword arguments.*"):
            TimeAgnosticGate('Q0', lambda **kwargs: op_1q)
        with pytest.raises(ValueError,
                           match=".*must be either Operation, or a func.*"):
            TimeAgnosticGate('Q0', 'Q1')

    def test_gate_params(self):
        angle_ref = -0.8435
        basis = (bases.general(2),)

        gate = TimeAgnosticGate('Q0', lambda angle: lib.rotate_y(angle))
        assert gate.qubits == ('Q0',)
        assert gate.params == {'angle'}
        ptm_ref = lib.rotate_y(angle=angle_ref).ptm(basis, basis)
        assert gate.operation(angle=angle_ref).ptm(basis, basis) == \
               approx(ptm_ref)
        assert gate.operation(angle=2 * angle_ref).ptm(basis, basis) != \
               approx(ptm_ref)

        kwargs = dict(angle=angle_ref, extra_param=42)
        assert gate.operation(**kwargs).ptm(basis, basis) == approx(ptm_ref)

        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"angle\" .*"):
            gate.operation()
        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"angle\" .*"):
            gate.operation(some_param=42, another_param=12)

        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib.rotate_y(angle_rotate).at(1),
                lib.cphase(angle_cphase).at(0, 1),
                lib.rotate_y(-angle_rotate).at(1))

        params = dict(angle_cphase=1.02 * pi, angle_rotate=0.47 * pi, foo='bar')
        ptm_ref = cnot_like(params['angle_cphase'], params['angle_rotate']) \
            .ptm(basis * 2, basis * 2)

        gate = TimeAgnosticGate(('D', 'A'), cnot_like)
        assert gate.qubits == ('D', 'A')
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate.operation(**params).ptm(basis * 2, basis * 2) == approx(
            ptm_ref)

        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"angle_cphase\" .*"):
            gate.operation(angle_rotate=0.99 * pi)

        with pytest.raises(RuntimeError,
                           match=".*number of qubits does not match .*"):
            TimeAgnosticGate(('Q0', 'Q1'), lambda phi: lib.rotate_z(phi)) \
                .operation(phi=angle_ref)

    def test_gate_params_set_number(self):
        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib.rotate_y(angle_rotate).at(1),
                lib.cphase(angle_cphase).at(0, 1),
                lib.rotate_y(-angle_rotate).at(1))

        angle_cphase_ref = 0.98 * pi
        angle_rotate_ref = 0.5 * pi
        basis = (bases.general(2),) * 2
        gate = TimeAgnosticGate(('D', 'A'), cnot_like)
        ptm_ref = gate.operation(angle_cphase=angle_cphase_ref,
                                 angle_rotate=angle_rotate_ref) \
            .ptm(basis, basis)

        assert gate.params == {'angle_cphase', 'angle_rotate'}
        gate.set(some_parameter=42)
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate.operation(angle_cphase=angle_cphase_ref,
                              angle_rotate=angle_rotate_ref) \
                   .ptm(basis, basis) == approx(ptm_ref)
        gate.set(angle_cphase=angle_cphase_ref)
        assert gate.params == {'angle_rotate'}
        assert gate.operation(angle_rotate=angle_rotate_ref) \
                   .ptm(basis, basis) == approx(ptm_ref)
        gate.set(angle_rotate=angle_rotate_ref, some_parameter=42)
        assert gate.params == set()
        assert gate.operation(another_parameter=12).ptm(basis, basis) == \
               approx(ptm_ref)

        gate = TimeAgnosticGate(('D', 'A'), cnot_like)
        gate.set(angle_rotate=angle_rotate_ref, angle_cphase=angle_cphase_ref,
                 some_parameter=42)
        assert gate.params == set()
        assert gate.operation().ptm(basis, basis) == approx(ptm_ref)

    def test_gate_params_rename(self):
        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib.rotate_y(angle_rotate).at(1),
                lib.cphase(angle_cphase).at(0, 1),
                lib.rotate_y(-angle_rotate).at(1))

        angle_cphase_ref = 0.98 * pi
        angle_rotate_ref = 0.5 * pi
        basis = (bases.general(2),) * 2
        gate = TimeAgnosticGate(('D', 'A'), cnot_like)
        ptm_ref = gate.operation(angle_cphase=angle_cphase_ref,
                                 angle_rotate=angle_rotate_ref) \
            .ptm(basis, basis)

        gate.set(angle_cphase='foo')
        assert gate.params == {'foo', 'angle_rotate'}
        assert gate.operation(foo=angle_cphase_ref,
                              angle_rotate=angle_rotate_ref) \
                   .ptm(basis, basis) == approx(ptm_ref)

        gate.set(foo='bar')
        assert gate.params == {'bar', 'angle_rotate'}
        assert gate.operation(bar=angle_cphase_ref,
                              angle_rotate=angle_rotate_ref) \
                   .ptm(basis, basis) == approx(ptm_ref)

        gate.set(bar=angle_cphase_ref)
        assert gate.params == {'angle_rotate'}
        assert gate.operation(angle_rotate=angle_rotate_ref,
                              angle_cphase=42.,
                              foo=12,
                              bar='and now something completely different') \
                   .ptm(basis, basis) == approx(ptm_ref)

        gate = TimeAgnosticGate(('D', 'A'), cnot_like)
        gate.set(angle_cphase='foo', angle_rotate='bar')
        assert gate.params == {'foo', 'bar'}
        assert gate.operation(bar=angle_rotate_ref,
                              foo=angle_cphase_ref,
                              angle_cphase=-1,
                              angle_rotate=-2) \
                   .ptm(basis, basis) == approx(ptm_ref)

        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"bar\" .*"):
            gate.operation(foo=0.99 * pi)
        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"bar\" .*"):
            gate.operation(foo=angle_rotate_ref, angle_cphase=angle_cphase_ref)
        with pytest.raises(ValueError,
                           match=".* not a valid Python identifier."):
            gate.set(bar='')
        with pytest.raises(ValueError,
                           match=".* not a valid Python identifier."):
            gate.set(bar='42')

    def test_gate_params_call(self):
        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib.rotate_y(angle_rotate).at(1),
                lib.cphase(angle_cphase).at(0, 1),
                lib.rotate_y(-angle_rotate).at(1))

        basis = (bases.general(2),) * 2
        angle1_cphase = 1.2*pi
        angle1_rotate = 0.6*pi
        angle2_cphase = 0.8*pi
        angle2_rotate = 0.3*pi

        gate = TimeAgnosticGate(('D', 'A'), cnot_like)
        gate1 = gate(angle_cphase=angle1_cphase, angle_rotate=angle1_rotate)
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate1.params == set()

        gate2 = gate(angle_cphase=angle2_cphase)
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate2.params == {'angle_rotate'}

        assert gate1.operation().ptm(basis, basis) == approx(gate.operation(
                angle_cphase=angle1_cphase, angle_rotate=angle1_rotate
            ).ptm(basis, basis))
        assert gate2.operation(angle_rotate=angle2_rotate).ptm(basis, basis) ==\
               approx(gate.operation(
                   angle_cphase=angle2_cphase, angle_rotate=angle2_rotate
               ).ptm(basis, basis))

    def test_circuits_add(self):
        orplus = lib.rotate_y(0.5 * pi)
        ocphase = lib.cphase(pi)
        orminus = lib.rotate_y(-0.5 * pi)
        grplus = TimeAgnosticGate('Q0', orplus)
        gcphase = TimeAgnosticGate(('Q0', 'Q1'), ocphase)
        grminus = TimeAgnosticGate('Q0', orminus)
        basis = (bases.general(2),) * 2

        circuit = grplus + gcphase
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 2
        assert circuit.operation().ptm(basis, basis) == approx(
            Operation.from_sequence(orplus.at(0), ocphase.at(0, 1))
            .ptm(basis, basis))

        circuit = circuit + grminus
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert circuit.operation().ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        circuit = grplus + (gcphase + grminus)
        assert circuit.qubits == ('Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert circuit.operation().ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        grplus = TimeAgnosticGate('Q1', orplus)
        grminus = TimeAgnosticGate('Q1', orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q1', 'Q0')
        assert len(circuit.gates) == 3
        assert circuit.operation().ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(0)
            ).ptm(basis, basis))

        basis = (basis[0],) * 3

        grplus = TimeAgnosticGate('Q2', orplus)
        grminus = TimeAgnosticGate('Q0', orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q2', 'Q0', 'Q1')
        assert len(circuit.gates) == 3
        assert circuit.operation().ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(1, 2), orminus.at(1)
            ).ptm(basis, basis))

        grplus = TimeAgnosticGate('Q0', orplus)
        grminus = TimeAgnosticGate('Q2', orminus)
        circuit = grplus + gcphase + grminus
        assert circuit.qubits == ('Q0', 'Q1', 'Q2')
        assert len(circuit.gates) == 3
        assert circuit.operation().ptm(basis, basis) == approx(
            Operation.from_sequence(
                orplus.at(0), ocphase.at(0, 1), orminus.at(2)
            ).ptm(basis, basis))

    def test_circuits_params(self):
        orotate = lib.rotate_y
        ocphase = lib.cphase
        grotate = TimeAgnosticGate('Q0', orotate)
        gcphase = TimeAgnosticGate(('Q0', 'Q1'), ocphase)
        basis = (bases.general(2),) * 2

        with pytest.raises(RuntimeError,
                           match=r".*free parameters.*\n"
                                 r".*angle.*\n"
                                 r".*allow_parameter_collisions.*"):
            gcphase + grotate

        with allow_parameter_collisions():
            circuit = grotate + gcphase + grotate

        assert circuit.params == {'angle'}
        assert len(circuit.gates) == 3
        angle = 0.736
        assert circuit.operation(angle=angle).ptm(basis, basis) == approx(
            Operation.from_sequence(
                orotate(angle).at(0), ocphase(angle).at(0, 1),
                orotate(angle).at(0)
            ).ptm(basis, basis))

        angle1 = 0.4*pi
        angle2 = 1.01*pi
        angle3 = -0.6*pi
        ptm_ref = Operation.from_sequence(
            orotate(angle1).at(0), ocphase(angle2).at(0, 1),
            orotate(angle3).at(0)
        ).ptm(basis, basis)

        circuit = grotate(angle=angle1) + gcphase(angle=angle2) + \
                  grotate(angle=angle3)
        assert circuit.operation().ptm(basis, basis) == approx(ptm_ref)

        circuit = grotate(angle='angle1') + gcphase(angle='angle2') + \
                  grotate(angle='angle3')
        assert circuit.operation(
            angle1=angle1, angle2=angle2, angle3=angle3
        ).ptm(basis, basis) == approx(ptm_ref)
