import pytest

from numpy import pi
from pytest import approx
import quantumsim.models.qubits as lib
from quantumsim.circuits import Gate
from quantumsim import bases, Operation


# noinspection PyTypeChecker
class TestCircuits:
    def test_untimed_gate_create_no_params(self):
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

    def test_untimed_gate_params(self):
        angle_ref = -0.8435
        basis = (bases.general(2),)

        gate = Gate('Q0', lambda angle: lib.rotate_y(angle))
        assert gate.qubits == ('Q0',)
        assert gate.params == ['angle']
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
        assert len(gate.params) == 2
        assert 'angle_cphase' in gate.params
        assert 'angle_rotate' in gate.params
        assert gate.operation(**params).ptm(basis*2, basis*2) == approx(ptm_ref)

        with pytest.raises(RuntimeError,
                           match="Can't construct .* \"angle_cphase\" .*"):
            gate.operation(angle_rotate=0.99*pi)

        with pytest.raises(RuntimeError,
                           match=".*number of qubits does not match .*"):
            Gate(('Q0', 'Q1'), lambda phi: lib.rotate_z(phi))\
                .operation(phi=angle_ref)
