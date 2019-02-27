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
from scipy.stats import unitary_group
from qs2.operations.algebra import kraus_to_ptm, single_kraus_to_ptm, ptm_convert_basis


@pytest.fixture(params=['numpy', 'cuda'])
def dm_class(request):
    mod = pytest.importorskip('qs2.states.' + request.param)
    return mod.DensityMatrix


@pytest.fixture(params=[[2]*9, [3, 2, 2]])
def dm_dims(request):
    return request.param


# FIXME: Gell-Mann should also be tested, when it is supported
# @pytest.fixture(params=[qs2.bases.general, qs2.bases.gell_mann])
@pytest.fixture(params=[qs2.bases.general])
def dm_basis(request):
    return request.param


def _basis_general_reshuffled():
    b0 = qs2.bases.general(2)
    order = (0, 2, 1, 3)
    vectors = np.array([b0.vectors[i] for i in order])
    labels = [b0.labels[i] for i in order]
    return qs2.bases.PauliBasis(vectors, labels)


basis_general_reshuffled = _basis_general_reshuffled()


def random_density_matrix(dim, seed):
    rng = np.random.RandomState(seed)
    diag = rng.rand(dim)
    diag /= np.sum(diag)
    dm = np.diag(diag)
    unitary = random_unitary_matrix(dim, seed+1)
    return unitary @ dm @ unitary.conj().T


def random_unitary_matrix(dim, seed):
    rng = np.random.RandomState(seed)
    return unitary_group.rvs(dim, random_state=rng)


class TestBackends:
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

    @pytest.mark.parametrize(
        'bases', [
            (qs2.bases.general(2), qs2.bases.general(2), qs2.bases.general(2)),
            (qs2.bases.general(2).subbasis([0, 1, 2]),
             qs2.bases.general(2).subbasis([0, 1]),
             qs2.bases.general(2))
        ])
    def test_create_from_dm_general(self, dm_class, bases):
        dm = np.zeros((8, 8), dtype=complex)
        dm[0, 0] = 0.25  # |000><000|
        dm[1, 1] = 0.75  # |001><001|
        dm[0, 1] = 0.5 - 0.33j  # |000><001|
        dm[1, 0] = 0.5 + 0.33j # |001><000|
        s = dm_class.from_dm(bases, dm)
        pv = s.expansion()
        assert pv.shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert s.bases[2] == bases[2]

        assert pv[0, 0, 0] == 0.25
        assert pv[0, 0, 1] == 0.75
        assert pv[0, 0, 2] == approx(0.5 * 2**0.5)
        assert pv[0, 0, 3] == approx(0.33 * 2**0.5)

    def test_create_from_random_dm(self, dm_class, dm_basis):
        dm = random_density_matrix(8, 34)
        bases = (dm_basis(2),) * 3
        s = dm_class.from_dm(bases, dm)
        assert s.expansion().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert s.bases[2] == bases[2]

        pv = s.expansion()
        dm2 = np.einsum('ijk,iad,jbe,kcf->abcdef',
                        pv, bases[0].vectors, bases[1].vectors,
                        bases[2].vectors, optimize=True).reshape(8, 8)
        assert dm2 == approx(dm)

    def test_apply_random_ptm1q_state1q(self, dm_class, dm_basis):
        unitary = random_unitary_matrix(2, 45)

        dm_before = random_density_matrix(2, seed=256)
        dm_after = unitary @ dm_before @ unitary.conj().T
        assert dm_before.trace() == approx(1)
        assert dm_after.trace() == approx(1)

        b = (dm_basis(2),)
        state0 = dm_class.from_dm(b, dm_before)
        state1 = dm_class.from_dm(b, dm_after)
        # sanity check
        assert state0.meas_prob(0) != approx(state1.meas_prob(0))

        ptm = single_kraus_to_ptm(unitary.reshape(1, 2, 2), b[0], b[0])
        b_gm = (qs2.bases.gell_mann(2),)
        ptm2 = ptm_convert_basis(ptm, b, b, b_gm, b_gm)
        assert np.allclose(ptm2[0, 1:], 0)
        assert ptm2[0, 0] == approx(1)

        state0.apply_ptm(ptm, 0)
        assert state0.expansion() == approx(state1.expansion())

    @pytest.mark.parametrize('qubit', [0, 1, 2])
    def test_apply_1q_random_ptm(self, dm_class, dm_basis, qubit):
        unity = np.identity(2)
        unitary = random_unitary_matrix(2, 45)
        einsum_args = [unity, [0, 3], unity, [1, 4], unity, [2, 5]]
        einsum_args[2*qubit] = unitary
        unitary3q = np.einsum(*einsum_args, optimize=True).reshape(8, 8)

        dm_before = random_density_matrix(8, seed=46)
        dm_after = unitary3q @ dm_before @ unitary3q.conj().T
        assert dm_before.trace() == approx(1)
        assert dm_after.trace() == approx(1)

        b = (dm_basis(2),)
        bases = b * 3
        state0 = dm_class.from_dm(bases, dm_before)
        state1 = dm_class.from_dm(bases, dm_after)
        # sanity check
        for q in range(3):
            if q != qubit:
                assert state0.meas_prob(q) == approx(state1.meas_prob(q))
            else:
                assert state0.meas_prob(q) != approx(state1.meas_prob(q))

        ptm = single_kraus_to_ptm(unitary.reshape(1, 2, 2), b[0], b[0])
        state0.apply_ptm(ptm, qubit)
        assert state0.expansion() == approx(state1.expansion())

    @pytest.mark.parametrize('qubits', [
        (0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)
    ])
    def test_apply_2q_random_ptm(self, dm_class, dm_basis, qubits):
        unity = np.identity(2)
        unitary = random_unitary_matrix(4, 45).reshape(2, 2, 2, 2)
        decay = np.array([[0.9, 0], [0, 0.7]])
        op = decay @ unitary

        q0, q1 = qubits
        (q2,) = tuple(q for q in range(3) if q not in qubits)
        einsum_args = [op, [q0, q1, q0+3, q1+3], unity, [q2, q2+3]]
        op3q = np.einsum(*einsum_args, optimize=True).reshape(8, 8)

        dm_before = random_density_matrix(8, seed=46)
        dm_after = op3q @ dm_before @ op3q.conj().T
        assert dm_before.trace() == approx(1)
        assert dm_after.trace() < 1.

        b = (dm_basis(2),)
        bases = b * 3
        state0 = dm_class.from_dm(bases, dm_before)
        state1 = dm_class.from_dm(bases, dm_after)

        ptm = kraus_to_ptm(op.reshape(1, 4, 4), b * 2, b * 2)
        state0.apply_ptm(ptm, *qubits)
        assert state0.expansion() == approx(state1.expansion())


    def test_project(self, dm_class, dm_basis, qubits):
        dm_before = random_density_matrix(8, seed=46)
        assert dm_before.trace() == approx(1)


    @pytest.mark.parametrize(
        'bases', [
            (qs2.bases.general(2), qs2.bases.general(2), qs2.bases.general(2)),
            (basis_general_reshuffled, basis_general_reshuffled,
             basis_general_reshuffled),
            (qs2.bases.general(2),
             qs2.bases.general(2).subbasis([0, 1]),
             qs2.bases.general(2).subbasis([0, 1, 2]))
        ])
    def test_diagonal_meas_prob(self, dm_class, bases):
        diag = np.array([0.25, 0, 0.75, 0, 0, 0, 0, 0])
        dm = np.diag(diag)
        s = dm_class.from_dm(bases, dm)
        assert s.expansion().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)
        assert s.trace() == approx(1)
        assert np.allclose(s.meas_prob(0), (1, 0))
        assert np.allclose(s.meas_prob(1), (0.25, 0.75))
        assert np.allclose(s.meas_prob(0), (1, 0))

        diag = np.array([0.25, 0.5, 0, 0, 0, 0, 0, 0])
        dm = np.diag(diag)
        s = dm_class.from_dm(bases, dm)
        assert s.expansion().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)
        assert s.trace() == approx(0.75)
        assert np.allclose(s.meas_prob(0), (0.75, 0.))
        assert np.allclose(s.meas_prob(1), (0.75, 0.))
        assert np.allclose(s.meas_prob(2), (0.25, 0.5))

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

    @pytest.mark.parametrize(
        'bases', [
            (qs2.bases.general(2), qs2.bases.general(2), qs2.bases.general(2)),
            (basis_general_reshuffled, basis_general_reshuffled,
             basis_general_reshuffled),
            (
                    qs2.bases.general(2).subbasis([0, 1]),
                    qs2.bases.general(2).subbasis([0, 1, 2]),
                    qs2.bases.general(2).subbasis([0, 1]),
            )
        ])
    def test_diagonal_indicated(self, dm_class, bases):
        dm = np.array([[min(i, j)*10 + max(i, j) for i in range(1, 9)]
                       for j in range(1, 9)])
        # dm = np.array([[min(i, j)*10 + max(i, j) for i in range(1, 5)]
        #                for j in range(1, 5)])
        diag = np.diag(dm)

        s = dm_class.from_dm(bases, dm)
        # assert s.expansion().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)

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
