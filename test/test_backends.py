import pytest
import qs2.bases
import numpy as np

from pytest import approx


@pytest.fixture(params=['numpy', 'cuda'])
def dm_class(request):
    mod = pytest.importorskip('qs2.backends.' + request.param)
    return mod.DensityMatrix


@pytest.fixture(params=[[2]*9, [3, 2, 2]])
def dm_dims(request):
    return request.param


# FIXME: Gell-Mann should also be tested, when it is supported
# @pytest.fixture(params=[qs2.bases.general, qs2.bases.gell_mann])
@pytest.fixture(params=[qs2.bases.general])
def dm_basis(request):
    return request.param


class TestBackends:
    def test_not_implemented_raised(self, dm_class):
        # Gell-Mann basis will certainly fail now
        with pytest.raises(NotImplementedError):
            dm = dm_class([qs2.bases.gell_mann(2)])

    def test_create_trivial(self, dm_class):
        dm = dm_class([])
        assert dm.expansion() == approx(1)
        assert dm.diagonal() == approx(1)

    def test_create(self, dm_class, dm_dims):
        target_shape = [dim**2 for dim in dm_dims]
        data = np.random.random_sample(target_shape)
        bases = [qs2.bases.general(dim) for dim in dm_dims]
        for expansion in (None, data):
            dm = dm_class(bases, expansion)
            assert dm.n_qubits == len(dm_dims)
            assert dm.dimensions == approx(dm_dims)
            assert dm.shape == approx(target_shape)
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

    @pytest.mark.parametrize('dim1,dim2', [[2, 2], [2, 3], [3, 3]])
    def test_get_diagonal(self, dm_class, dm_basis, dim1, dim2):
        basis1 = dm_basis(dim1)
        basis2 = dm_basis(dim2)

        # Default initialization
        dm = dm_class([basis1, basis2])
        diag = dm.diagonal()
        diag_ref = np.zeros(dim1*dim2)
        diag_ref[0] = 1.
        assert diag == approx(diag_ref)
