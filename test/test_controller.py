import pytest
import numpy as np
import xarray as xr
from pytest import approx

from quantumsim import Controller, gates, State


# noinspection PyTypeChecker
class TestController:
    def test_controller_init(self):
        with pytest.raises(ValueError):
            c = Controller(None)
        with pytest.raises(ValueError):
            c = Controller({})

        circ = (gates.rotate_x('Q0') + gates.cnot('Q0', 'Q1'))
        circ_f = circ.finalize()
        with pytest.raises(ValueError):
            c = Controller(circ)

        with pytest.raises(ValueError):
            c = Controller({None, circ})

        with pytest.raises(ValueError):
            c = Controller(dict(test=circ))

        c = Controller(dict(test=circ_f))
        assert c._qubits == set(['Q0', 'Q1'])
        assert c._parameters == {}

    def test_prepare_state(self):
        circ = (gates.rotate_x('Q0') + gates.cnot('Q0', 'Q1'))
        circ_f = circ.finalize()
        c = Controller(dict(test=circ_f))
        assert c.state is None
        c.prepare_state()
        assert c.state is not None
        assert isinstance(c.state, State)
        assert np.allclose(c.state.pauli_vector.to_dm(),
                           State(["Q0", "Q1"]).pauli_vector.to_dm())

        c = Controller(dict(test=circ_f))
        c.prepare_state(dim=3)
        assert np.allclose(c.state.pauli_vector.to_dm(),
                           State(["Q0", "Q1"], dim=3).pauli_vector.to_dm())
