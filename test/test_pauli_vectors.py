# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

"""Purpose of the tests in this file is mostly to check, that all interface
methods in the backends can be called and exceptions are raised,
where necessary.
"""
import pytest
import quantumsim.bases
import numpy as np

from pytest import approx
from scipy.stats import unitary_group
from quantumsim.algebra import kraus_to_ptm, ptm_convert_basis


@pytest.fixture(params=[
    ('quantumsim.pauli_vectors.numpy', 'PauliVectorNumpy'),
    ('quantumsim.pauli_vectors.cuda', 'PauliVectorCuda'),
])
def pauli_vector_cls(request):
    mod = pytest.importorskip(request.param[0])
    return getattr(mod, request.param[1])


@pytest.fixture(params=[[2]*9, [3, 2, 2]])
def dm_dims(request):
    return request.param


# FIXME: Gell-Mann should also be tested, when it is supported
# @pytest.fixture(params=[quantumsim.bases.general, quantumsim.bases.gell_mann])
@pytest.fixture(params=[quantumsim.bases.general])
def dm_basis(request):
    return request.param


def _basis_general_reshuffled():
    b0 = quantumsim.bases.general(2)
    order = (0, 2, 1, 3)
    vectors = np.array([b0.vectors[i] for i in order])
    labels = [b0.labels[i] for i in order]
    return quantumsim.bases.PauliBasis(vectors, labels)


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


class TestPauliVectors:
    def test_create_trivial(self, pauli_vector_cls):
        dm = pauli_vector_cls([])
        assert dm.to_pv() == approx(1)
        assert dm.diagonal() == approx(1)

    def test_create(self, pauli_vector_cls, dm_dims):
        target_shape = [dim**2 for dim in dm_dims]
        data = np.random.random_sample(target_shape)
        bases = [quantumsim.bases.general(dim) for dim in dm_dims]
        for expansion in (None, data):
            dm = pauli_vector_cls(bases, expansion)
            assert dm.n_qubits == len(dm_dims)
            assert dm.dim_hilbert == approx(dm_dims)
            assert dm.dim_pauli == approx(target_shape)
            assert dm.size == np.product(np.array(dm_dims)**2)
            if expansion is not None:
                np.testing.assert_almost_equal(dm.to_pv(), expansion)
            del dm

        # wrong data dimensionality
        wrong_dim = np.product(np.array(dm_dims))
        wrong_shape = (wrong_dim, wrong_dim)
        data = data.reshape(wrong_shape)
        with pytest.raises(ValueError):
            pauli_vector_cls(bases, data)

        # we require expansion in basis, it should be float
        for shape in (target_shape, wrong_shape):
            data = np.zeros(shape, dtype=complex)
            with pytest.raises(ValueError):
                pauli_vector_cls(bases=bases, pv=data)

        # 16 qubits is too much
        with pytest.raises(ValueError):
            pauli_vector_cls(bases=[quantumsim.bases.general(2)] * 16)

    @pytest.mark.parametrize(
        'bases', [
            (quantumsim.bases.general(2), quantumsim.bases.general(2), quantumsim.bases.general(2)),
            (quantumsim.bases.general(2).subbasis([0, 1, 2]),
             quantumsim.bases.general(2).subbasis([0, 1]),
             quantumsim.bases.general(2))
        ])
    def test_create_from_dm_general(self, pauli_vector_cls, bases):
        dm = np.zeros((8, 8), dtype=complex)
        dm[0, 0] = 0.25  # |000><000|
        dm[1, 1] = 0.75  # |001><001|
        dm[0, 1] = 0.5 - 0.33j  # |000><001|
        dm[1, 0] = 0.5 + 0.33j  # |001><000|
        s = pauli_vector_cls.from_dm(dm, bases)
        pv = s.to_pv()
        assert pv.shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert s.bases[2] == bases[2]

        assert pv[0, 0, 0] == 0.25
        assert pv[0, 0, 1] == 0.75
        assert pv[0, 0, 2] == approx(0.5 * 2**0.5)
        assert pv[0, 0, 3] == approx(0.33 * 2**0.5)

    def test_create_from_random_dm(self, pauli_vector_cls, dm_basis):
        dm = random_density_matrix(8, 34)
        bases = (dm_basis(2),) * 3
        s = pauli_vector_cls.from_dm(dm, bases)
        assert s.to_pv().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert s.bases[2] == bases[2]

        pv = s.to_pv()
        dm2 = np.einsum('ijk,iad,jbe,kcf->abcdef',
                        pv, bases[0].vectors, bases[1].vectors,
                        bases[2].vectors, optimize=True).reshape(8, 8)
        assert dm2 == approx(dm)

    def test_apply_random_ptm1q_pv1q(self, pauli_vector_cls, dm_basis):
        unitary = random_unitary_matrix(2, 45)

        dm_before = random_density_matrix(2, seed=256)
        dm_after = unitary @ dm_before @ unitary.conj().T
        assert dm_before.trace() == approx(1)
        assert dm_after.trace() == approx(1)

        b = (dm_basis(2),)
        pv0 = pauli_vector_cls.from_dm(dm_before, b)
        pv1 = pauli_vector_cls.from_dm(dm_after, b)
        # sanity check
        assert pv0.meas_prob(0) != approx(pv1.meas_prob(0))

        ptm = kraus_to_ptm(unitary.reshape(1, 2, 2), b, b)
        b_gm = (quantumsim.bases.gell_mann(2),)
        ptm2 = ptm_convert_basis(ptm, b, b, b_gm, b_gm)
        assert np.allclose(ptm2[0, 1:], 0)
        assert ptm2[0, 0] == approx(1)

        pv0.apply_ptm(ptm, 0)
        assert pv0.to_pv() == approx(pv1.to_pv())

    @pytest.mark.parametrize('qubit', [0, 1, 2])
    def test_apply_1q_random_ptm(self, pauli_vector_cls, dm_basis, qubit):
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
        pv0 = pauli_vector_cls.from_dm(dm_before, bases)
        pv1 = pauli_vector_cls.from_dm(dm_after, bases)
        # sanity check
        for q in range(3):
            if q != qubit:
                assert pv0.meas_prob(q) == approx(pv1.meas_prob(q))
            else:
                assert pv0.meas_prob(q) != approx(pv1.meas_prob(q))

        ptm = kraus_to_ptm(unitary.reshape(1, 2, 2), b, b)
        pv0.apply_ptm(ptm, qubit)
        assert pv0.to_pv() == approx(pv1.to_pv())

    @pytest.mark.parametrize('qubits', [
        (0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)
    ])
    def test_apply_2q_random_ptm(self, pauli_vector_cls, dm_basis, qubits):
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
        pv0 = pauli_vector_cls.from_dm(dm_before, bases)
        pv1 = pauli_vector_cls.from_dm(dm_after, bases)

        ptm = kraus_to_ptm(op.reshape(1, 4, 4), b * 2, b * 2)
        pv0.apply_ptm(ptm, *qubits)
        assert pv0.to_pv() == approx(pv1.to_pv())

    @pytest.mark.parametrize(
        'bases', [
            (quantumsim.bases.general(2), quantumsim.bases.general(2), quantumsim.bases.general(2)),
            (basis_general_reshuffled, basis_general_reshuffled,
             basis_general_reshuffled),
            (quantumsim.bases.general(2),
             quantumsim.bases.general(2).subbasis([0, 1]),
             quantumsim.bases.general(2).subbasis([0, 1, 2]))
        ])
    def test_diagonal_meas_prob(self, pauli_vector_cls, bases):
        diag = np.array([0.25, 0, 0.75, 0, 0, 0, 0, 0])
        dm = np.diag(diag)
        s = pauli_vector_cls.from_dm(dm, bases)
        assert s.to_pv().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)
        assert s.trace() == approx(1)
        assert np.allclose(s.meas_prob(0), (1, 0))
        assert np.allclose(s.meas_prob(1), (0.25, 0.75))
        assert np.allclose(s.meas_prob(0), (1, 0))

        diag = np.array([0.25, 0.5, 0, 0, 0, 0, 0, 0])
        dm = np.diag(diag)
        s = pauli_vector_cls.from_dm(dm, bases)
        assert s.to_pv().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)
        assert s.trace() == approx(0.75)
        assert np.allclose(s.meas_prob(0), (0.75, 0.))
        assert np.allclose(s.meas_prob(1), (0.75, 0.))
        assert np.allclose(s.meas_prob(2), (0.25, 0.5))

    @pytest.mark.parametrize('dim1,dim2', [[2, 2], [2, 3], [3, 3]])
    def test_get_diagonal(self, pauli_vector_cls, dm_basis, dim1, dim2):
        basis1 = dm_basis(dim1)
        basis2 = dm_basis(dim2)

        # Default initialization
        dm = pauli_vector_cls([basis1, basis2])
        diag = dm.diagonal()
        diag_ref = np.zeros(dim1*dim2)
        diag_ref[0] = 1.
        assert diag == approx(diag_ref)

    @pytest.mark.parametrize(
        'bases', [
            (quantumsim.bases.general(2), quantumsim.bases.general(2), quantumsim.bases.general(2)),
            (basis_general_reshuffled, basis_general_reshuffled,
             basis_general_reshuffled),
            (
                    quantumsim.bases.general(2).subbasis([0, 1]),
                    quantumsim.bases.general(2).subbasis([0, 1, 2]),
                    quantumsim.bases.general(2).subbasis([0, 1]),
            )
        ])
    def test_diagonal_indicated(self, pauli_vector_cls, bases):
        dm = np.array([[min(i, j)*10 + max(i, j) for i in range(1, 9)]
                       for j in range(1, 9)])
        # dm = np.array([[min(i, j)*10 + max(i, j) for i in range(1, 5)]
        #                for j in range(1, 5)])
        diag = np.diag(dm)

        s = pauli_vector_cls.from_dm(dm, bases)
        # assert s.expansion().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)

