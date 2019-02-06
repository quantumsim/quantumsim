# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

"""Purpose of the tests in this file is mostly to check, that all interface
methods in the bachends can be called and exceptions are raised,
where necessary. Result sanity validation are done in test_state.py and
test_operations.py
"""
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
            dm_class([qs2.bases.gell_mann(2)])

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
            assert dm.dim_hilbert == approx(dm_dims)
            assert dm.dim_pauli == approx(target_shape)
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

    def test_add_qubit(self, dm_class, dm_basis):
        dm = dm_class([])
        bases = []
        expected_dim_pauli = []
        assert dm.n_qubits == 0
        assert dm.dim_pauli == approx(expected_dim_pauli)

        for dim in (2, 2, 3):
            basis = dm_basis(dim)
            bases.append(basis)
            expected_dim_pauli.insert(0, dim**2)
            dm.add_qubit(basis, 0)
            assert dm.n_qubits == len(bases)
            assert dm.dim_pauli == approx(expected_dim_pauli)
            assert dm.expansion().shape == approx(expected_dim_pauli)

    def test_project(self, dm_class, dm_basis):
        # matrix in |0, 0, 0> state
        dim_hilbert = [2, 2, 3]
        dm = dm_class([dm_basis(d) for d in dim_hilbert])
        assert dm.trace() == 1
        assert dm.dim_pauli == approx([4, 4, 9])
        dm.project(0, 0)
        data = dm.expansion().copy()
        assert dm.dim_pauli == approx([1, 4, 9])
        assert dm.trace() == 1
        dm.project(0, 0)  # should do nothing
        assert dm.expansion() == approx(data)
        assert dm.dim_pauli == approx([1, 4, 9])
        with pytest.raises(RuntimeError):
            dm.project(0, 1)

        # matrix in 0.8 |0, 0, 0> + 0.2 |1, 1, 0> state
        dim_hilbert = [3, 2, 2]
        bases = [dm_basis(d) for d in dim_hilbert]
        data = np.zeros([b.dim_pauli for b in bases], dtype=float)
        data[0, 0, 0] = 0.8
        data[1, 1, 0] = 0.2
        dm = dm_class(bases, data)
        assert dm.trace() == approx(1)
        assert dm.dim_pauli == approx([9, 4, 4])
        dm.project(0, 0)
        assert dm.trace() == approx(0.8)
        assert dm.dim_pauli == approx([1, 4, 4])
        dm.project(2, 0)
        assert dm.trace() == approx(0.8)
        assert dm.dim_pauli == approx([1, 4, 1])
        dm.project(1, 1)
        assert dm.trace() == approx(0.)
        assert dm.dim_pauli == approx([1, 1, 1])
