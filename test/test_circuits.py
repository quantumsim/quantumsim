import pytest

from numpy import pi
from pytest import approx
import quantumsim.models.qubits as lib
from quantumsim.circuits import Gate
from quantumsim import bases, Operation


# noinspection PyTypeChecker
class TestCircuits:
    def test_gate_create_no_params(self):
        angle = 1.0758

        op_1q = lib.rotate_x(angle)
        op_2q = lib.cphase(angle)

        gate = Gate('qubit', op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('qubit',)
        assert len(gate.params) == 0

        gate = Gate('Q0', lambda: op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('Q0',)
        assert len(gate.params) == 0

        gate = Gate(('Q1',), lambda: op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('Q1',)
        assert len(gate.params) == 0

        gate = Gate(('D', 'A'), lambda: op_2q)
        assert gate.operation() == op_2q
        assert gate.qubits == ('D', 'A')
        assert len(gate.params) == 0

        with pytest.raises(RuntimeError,
                           match=".*number of qubits does not match .*"):
            Gate('Q0', op_2q).operation()
        with pytest.raises(RuntimeError,
                           match=".*number of qubits does not match .*"):
            Gate(('Q0', 'Q1'), lambda: op_1q).operation()
        with pytest.raises(RuntimeError,
                           match="Invalid operation function was provided.*"):
            Gate('Q0', lambda: [[1., 0.], [0., 1.]]).operation()

        with pytest.raises(ValueError,
                           match=".*can't accept free arguments.*"):
            Gate('Q0', lambda *args: op_1q)
        with pytest.raises(ValueError,
                           match=".*can't accept free keyword arguments.*"):
            Gate('Q0', lambda **kwargs: op_1q)
        with pytest.raises(ValueError,
                           match=".*must be either Operation, or a func.*"):
            Gate('Q0', 'Q1')

    def test_gate_params(self):
        angle_ref = -0.8435
        basis = (bases.general(2),)

        gate = Gate('Q0', lambda angle: lib.rotate_y(angle))
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

        params = dict(angle_cphase=1.02*pi, angle_rotate=0.47*pi, foo='bar')
        ptm_ref = cnot_like(params['angle_cphase'], params['angle_rotate'])\
            .ptm(basis*2, basis*2)

        gate = Gate(('D', 'A'), cnot_like)
        assert gate.qubits == ('D', 'A')
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate.operation(**params).ptm(basis*2, basis*2) == approx(ptm_ref)

        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"angle_cphase\" .*"):
            gate.operation(angle_rotate=0.99*pi)

        with pytest.raises(RuntimeError,
                           match=".*number of qubits does not match .*"):
            Gate(('Q0', 'Q1'), lambda phi: lib.rotate_z(phi))\
                .operation(phi=angle_ref)

    def test_gate_params_set_number(self):
        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib.rotate_y(angle_rotate).at(1),
                lib.cphase(angle_cphase).at(0, 1),
                lib.rotate_y(-angle_rotate).at(1))

        angle_cphase_ref = 0.98*pi
        angle_rotate_ref = 0.5*pi
        basis = (bases.general(2),) * 2
        gate = Gate(('D', 'A'), cnot_like)
        ptm_ref = gate.operation(angle_cphase=angle_cphase_ref,
                                 angle_rotate=angle_rotate_ref)\
                      .ptm(basis, basis)

        assert gate.params == {'angle_cphase', 'angle_rotate'}
        gate.set(some_parameter=42)
        assert gate.params == {'angle_cphase', 'angle_rotate'}
        assert gate.operation(angle_cphase=angle_cphase_ref,
                              angle_rotate=angle_rotate_ref) \
            .ptm(basis, basis) == approx(ptm_ref)
        gate.set(angle_cphase=angle_cphase_ref)
        assert gate.params == {'angle_rotate'}
        assert gate.operation(angle_rotate=angle_rotate_ref)\
                   .ptm(basis, basis) == approx(ptm_ref)
        gate.set(angle_rotate=angle_rotate_ref, some_parameter=42)
        assert gate.params == set()
        assert gate.operation(another_parameter=12).ptm(basis, basis) == \
               approx(ptm_ref)

        gate = Gate(('D', 'A'), cnot_like)
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

        angle_cphase_ref = 0.98*pi
        angle_rotate_ref = 0.5*pi
        basis = (bases.general(2),) * 2
        gate = Gate(('D', 'A'), cnot_like)
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

        gate = Gate(('D', 'A'), cnot_like)
        gate.set(angle_cphase='foo', angle_rotate='bar')
        assert gate.params == {'foo', 'bar'}
        assert gate.operation(bar=angle_rotate_ref,
                              foo=angle_cphase_ref,
                              angle_cphase=-1,
                              angle_rotate=-2) \
                   .ptm(basis, basis) == approx(ptm_ref)

        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"bar\" .*"):
            gate.operation(foo=0.99*pi)
        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"bar\" .*"):
            gate.operation(foo=angle_rotate_ref, angle_cphase=angle_cphase_ref)
        with pytest.raises(ValueError,
                           match=".* not a valid Python identifier."):
            gate.set(bar='')
        with pytest.raises(ValueError,
                           match=".* not a valid Python identifier."):
            gate.set(bar='42')
