import pytest

import quantumsim.models.qubits as lib
from quantumsim.circuits import Gate


# noinspection PyTypeChecker
class TestCircuits:
    def test_untimed_gate_create_no_params(self):
        angle = 1.0743

        op_1q = lib.rotate_x(angle)
        op_2q = lib.cphase(angle)

        gate = Gate('qubit', op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('qubit',)

        gate = Gate('Q0', lambda: op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('Q0',)

        gate = Gate(('Q1',), lambda: op_1q)
        assert gate.operation() == op_1q
        assert gate.qubits == ('Q1',)

        gate = Gate(('D', 'A'), lambda: op_2q)
        assert gate.operation() == op_2q
        assert gate.qubits == ('D', 'A')

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
