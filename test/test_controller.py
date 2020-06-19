import pytest
import numpy as np
import xarray as xr
from pytest import approx

from quantumsim import Controller, gates, State


# noinspection PyTypeChecker
class TestController:
    def test_controller_init(self):
        with pytest.raises(TypeError):
            c = Controller(None)

        circ = (gates.rotate_x('Q0') + gates.cnot('Q0', 'Q1'))
        final_circ = circ.finalize()

        with pytest.raises(TypeError):
            c = Controller(circ)
        with pytest.raises(TypeError):
            c = Controller([circ, final_circ])
        with pytest.raises(ValueError):
            c = Controller({})

        with pytest.raises(TypeError):
            c = Controller({None, circ})

        with pytest.raises(TypeError):
            c = Controller(dict(test=circ))

        c = Controller(dict(test=final_circ))
        assert c._qubits == set(['Q0', 'Q1'])
        assert c._parameters == {}

    def test_prepare_state(self):
        circ = (gates.rotate_x('Q0') + gates.cnot('Q0', 'Q1')).finalize()
        c = Controller(dict(test=circ))
        assert c.state is None
        c.prepare_state()
        assert c.state is not None
        assert isinstance(c.state, State)
        assert np.allclose(c.state.pauli_vector.to_dm(),
                           State(["Q0", "Q1"]).pauli_vector.to_dm())

        c = Controller(dict(test=circ))
        c.prepare_state(dim=3)
        assert np.allclose(c.state.pauli_vector.to_dm(),
                           State(["Q0", "Q1"], dim=3).pauli_vector.to_dm())

    def test_set_rng(self):
        c = Controller(dict(test=gates.rotate_x('Q0').finalize()))
        assert c._rng is None

        with pytest.raises(TypeError):
            c.set_rng('string')

        with pytest.raises(TypeError):
            c.set_rng([1, 2, 3])

        with pytest.raises(TypeError):
            c.set_rng(np.random.RandomState(42))

        c.set_rng(42)
        assert isinstance(c._rng, np.random.RandomState)

    def test_apply_circuit(self):
        circ = (gates.rotate_x('Q0', angle=90)).finalize()
        rng_circ = (gates.measure('Q0')).finalize()

        c = Controller(dict(
            test=circ,
            rng_test=rng_circ))

        c.prepare_state()
        c.set_rng(42)

        out = c._apply_circuit(circ)
        assert out is None
        meas_ps = c.state.pauli_vector.meas_prob(0)
        assert all(prob == approx(0.5) for prob in meas_ps)

        out = c._apply_circuit(rng_circ,
                               param_funcs={'result': lambda state: 1})
        assert out is not None
        assert isinstance(out, xr.DataArray)
        coords = list(out.coords)
        assert len(coords) == 1
        assert 'param' in coords
        assert 'result' in out.param
        c.state.renormalize()
        meas_ps = c.state.pauli_vector.meas_prob(0)
        assert meas_ps[0] == approx(0)
        assert meas_ps[1] == approx(1)

    def test_apply(self):
        circ = (gates.rotate_x('Q0') + gates.cnot('Q0', 'Q1')).finalize()
        rng_circ = (gates.measure('Q0')).finalize()
        rng_param_circ = (gates.rotate_x('Q0', angle='ang') +
                          gates.measure('Q0', result='res')).finalize()
        c = Controller(dict(
            test=circ,
            rng_test=rng_circ,
            rng_param_test=rng_param_circ))
        with pytest.raises(KeyError):
            c.apply("unknown_circ")

        with pytest.raises(ValueError):
            c.apply('test')

        c.prepare_state()
        with pytest.raises(KeyError):
            c.apply('test')

        out = c.apply('test', angle=90)
        assert out is None

        with pytest.raises(AttributeError):
            c.apply('rng_test')

        c.set_rng(42)
        out = c.apply('rng_test')

        assert isinstance(out, xr.DataArray)
        assert out.name == 'rng_test'
        assert not out.attrs
        assert 'param' in out.coords
        assert out.shape == (1,)
        assert list(out.param.values) == ['result']

        out = c.apply('rng_test', cycle=1)
        assert 'cycle' in out.coords

        out = c.apply('rng_param_test', cycle=1, ang=90)
        assert 'cycle' in out.coords
        assert 'ang' not in out.coords

        param_c = Controller(dict(rng_param_test=rng_param_circ),
                             parameters=dict(ang=lambda outcome: 180 if outcome.sel(param='res') == 1 else 0))
        param_c.prepare_state()
        param_c.set_rng(42)
        out = param_c.apply('rng_param_test')

        assert isinstance(out, xr.DataArray)
        assert out.shape == (2,)
        assert all(p in out.param for p in ['res', 'ang'])

    def test_to_dataset(self):
        circ = (gates.rotate_x('Q0') + gates.cnot('Q0', 'Q1')).finalize()
        c = Controller(dict(test=circ))

        with pytest.raises(TypeError):
            c.to_dataset(np.array([1, 2]))

        with pytest.raises(TypeError):
            c.to_dataset([0, 1])

        da = xr.DataArray(data=[0, 1, 2], dims=['ind'])
        with pytest.raises(ValueError):
            c.to_dataset(da)

        da.name = 'test'
        c.to_dataset(da)

        assert c._outcomes
        assert 'test' in c._outcomes
        assert isinstance(c._outcomes['test'], list)
        assert isinstance(c._outcomes['test'][0], xr.DataArray)
        assert all(c._outcomes['test'][0] == da)
        assert c._outcomes['test'][0].concat_dim is None

        c.to_dataset(da, concat_dim='dim')
        assert len(c._outcomes['test']) == 2
        assert c._outcomes['test'][1].concat_dim == 'dim'

    def test_get_dataset(self):
        circ = (gates.rotate_x('Q0') + gates.cnot('Q0', 'Q1')).finalize()
        rng_circ = (gates.measure('Q0')).finalize()

        c = Controller(dict(
            test=circ,
            rng_test=rng_circ))

        c.prepare_state()
        c.set_rng(42)

        ds = c.get_dataset()
        assert ds is None

        c.to_dataset(c.apply('test', angle=90))
        ds = c.get_dataset()
        assert ds is None

        c.to_dataset(c.apply('rng_test'))
        ds = c.get_dataset()
        assert ds is not None
        assert isinstance(ds, xr.Dataset)
        data_vars = list(ds.data_vars)
        assert len(data_vars) == 1
        assert 'rng_test' in data_vars
        coords = list(ds.coords)
        assert len(coords) == 1
        assert 'param' in coords

    def test_run(self):
        circ = (gates.rotate_x('Q0', angle=180) +
                gates.measure('Q0', result='meas_out') +
                gates.rotate_x('Q0', angle='cond_angle')).finalize()

        class SampleController(Controller):
            def sample_exp(self):
                self.prepare_state()

                for cycle in range(2):
                    self.to_dataset(self.apply(
                        'circ', cycle=cycle), concat_dim='cycle')

        controller = SampleController(
            circuits={
                'circ': circ,
            },
            parameters={
                'cond_angle': lambda outcome: 180 if outcome.sel(param='meas_out') == 1 else 0,
            }
        )

        outcome = controller.run(controller.sample_exp, seed=range(3))
        assert outcome is not None
        assert isinstance(outcome, xr.Dataset)

        assert len(outcome) == 1
        assert 'circ' in outcome

        assert len(outcome.coords) == 3
        assert all(cord in outcome.coords for cord in [
                   'cycle', 'param', 'seed'])

        assert len(outcome.param) == 2
        assert all(par in outcome.param for par in ['meas_out', 'cond_angle'])
        assert all(outcome.cycle == range(2))
        assert all(outcome.seed == range(3))
        assert np.all([outcome.circ.sel(param='meas_out') == 1])
        assert np.all([outcome.circ.sel(param='cond_angle') == 180.0])
