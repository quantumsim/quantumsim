import pytest
import qs2.bases
import numpy as np


@pytest.fixture(params=['numpy', 'cuda'])
def dm_class(request):
    mod = pytest.importorskip('qs2.backends.' + request.param)
    return mod.DensityMatrix


@pytest.fixture(params=[[2]*9, [3, 2, 2]])
def dm_dims(request):
    return request.param


class TestBackends:

    def test_create_trivial(self, dm_class):
        dm = dm_class([])
        assert dm.expansion() == pytest.approx(1)
        assert dm.diagonal() == pytest.approx(1)

    def test_create(self, dm_class, dm_dims):
        target_shape = [dim**2 for dim in dm_dims]
        data = np.random.random_sample(target_shape)
        bases = [qs2.bases.general(dim) for dim in dm_dims]
        for expansion in (None, data):
            dm = dm_class(bases, expansion)
            assert dm.n_qubits == len(dm_dims)
            np.testing.assert_array_equal(dm.dimensions, dm_dims)
            np.testing.assert_array_equal(dm.shape, target_shape)
            assert dm.size == np.product(np.array(dm_dims)**2)
            if expansion is not None:
                np.testing.assert_almost_equal(dm.expansion(), expansion)
            del dm

        # wrong data dimensionality
        wrong_dim = np.product(np.array(dm_dims))
        wrong_shape = (wrong_dim, wrong_dim)
        data = data.reshape(wrong_shape)
        with pytest.raises(ValueError):
            dm_class(bases, data)

        # we require expansion in basis, it should be float
        for shape in (target_shape, wrong_shape):
            data = np.zeros(shape, dtype=complex)
            with pytest.raises(ValueError):
                dm_class(bases=bases, expansion=data)

        # 16 qubits is too much
        with pytest.raises(ValueError):
            dm_class(bases=[qs2.bases.general(2)]*16)



