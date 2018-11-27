import pytest
import qs2.bases


@pytest.fixture(params=['numpy', 'cuda'])
def dm_class(request):
    mod = pytest.importorskip('qs2.backends.' + request.param)
    return mod.DensityMatrix


class TestBackends:
    @pytest.mark.parametrize("dims", [[2]*9, [3, 2, 2]])
    def test_create(self, dm_class, dims):
        bases = [qs2.bases.general(dim) for dim in dims]
        dm = dm_class(bases)
        assert dm.n_qubits == len(dims)
