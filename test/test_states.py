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
from quantumsim.algebra import kraus_to_ptm, ptm_convert_basis
from quantumsim.algebra.tools import random_hermitian_matrix, random_unitary_matrix


@pytest.fixture(params=[
    ('quantumsim.states.numpy', 'StateNumpy'),
    ('quantumsim.states.cuda', 'StateCuda'),
])
def state_cls(request):
    mod = pytest.importorskip(request.param[0])
    return getattr(mod, request.param[1])


@pytest.fixture(params=[2, 3])
def dim_hilbert(request):
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


class TestStates:
    def test_create_common(self, state_cls):
        dm = state_cls([])
        assert dm.to_pv() == approx(1)
        assert dm.diagonal() == approx(1)

        # bases must have the same Hilbert dimensionality
        bases = (quantumsim.bases.general(2), quantumsim.bases.general(3))
        pauli_vector = np.random.random_sample((4, 9))
        with pytest.raises(ValueError, match='All basis elements must have the same '
                                             'Hilbert dimensionality'):
            state_cls(2, pauli_vector, bases)

    def test_create(self, state_cls, dim_hilbert):
        num_qubits = 4
        target_shape = (dim_hilbert**2,) * num_qubits
        pauli_vector = np.random.random_sample(target_shape)
        bases = [quantumsim.bases.general(dim_hilbert)] * num_qubits

        state = state_cls(num_qubits, dim_hilbert=dim_hilbert)
        assert state.qubits == list(range(num_qubits))
        assert state.dim_hilbert == dim_hilbert
        assert state.dim_pauli == (1,) * num_qubits
        assert state.size == 1
        del state

        qubits = [f'q{i}' for i in range(num_qubits)]
        state = state_cls(qubits, pauli_vector, bases)
        assert state.qubits == qubits
        assert state.dim_hilbert == dim_hilbert
        assert state.dim_pauli == target_shape
        assert state.size == (dim_hilbert ** 2) ** num_qubits
        assert np.allclose(state.to_pv(), pauli_vector)
        del state

        with pytest.raises(ValueError, match='Both `pv` and `bases` must be provided'):
            state_cls(qubits, pauli_vector)
        with pytest.raises(ValueError, match='Both `pv` and `bases` must be provided'):
            state_cls(num_qubits, bases=bases)

        # wrong data dimensionality
        wrong_dim = dim_hilbert ** num_qubits
        wrong_shape = (wrong_dim, wrong_dim)
        pauli_vector = pauli_vector.reshape(wrong_shape)
        with pytest.raises(ValueError):
            state_cls(qubits, pauli_vector, bases)

        # we require expansion in basis, it should be float
        pauli_vector = np.zeros(target_shape, dtype=complex)
        with pytest.raises(ValueError):
            state_cls(num_qubits, bases=bases, pv=pauli_vector)

    @pytest.mark.parametrize(
        'bases', [(quantumsim.bases.general(2),
                   quantumsim.bases.general(2),
                   quantumsim.bases.general(2)),
                  (quantumsim.bases.general(2).subbasis([0, 1, 2]),
                   quantumsim.bases.general(2).subbasis([0, 1]),
                   quantumsim.bases.general(2))])
    def test_create_from_dm_general(self, state_cls, bases):
        dm = np.zeros((8, 8), dtype=complex)
        dm[0, 0] = 0.25  # |000><000|
        dm[1, 1] = 0.75  # |001><001|
        dm[0, 1] = 0.5 - 0.33j  # |000><001|
        dm[1, 0] = 0.5 + 0.33j  # |001><000|
        state = state_cls.from_dm(dm, bases)
        pv = state.to_pv()
        assert pv.shape == tuple(b.dim_pauli for b in bases)
        assert state.bases[0] == bases[0]
        assert state.bases[1] == bases[1]
        assert state.bases[2] == bases[2]

        assert pv[0, 0, 0] == 0.25
        assert pv[0, 0, 1] == 0.75
        assert pv[0, 0, 2] == approx(0.5 * 2**0.5)
        assert pv[0, 0, 3] == approx(0.33 * 2**0.5)

    def test_create_from_random_dm(self, state_cls, dm_basis):
        dm = random_hermitian_matrix(8, 34)
        bases = (dm_basis(2),) * 3
        s = state_cls.from_dm(dm, bases)
        assert s.to_pv().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert s.bases[2] == bases[2]

        pv = s.to_pv()
        dm2 = np.einsum('ijk,iad,jbe,kcf->abcdef',
                        pv, bases[0].vectors, bases[1].vectors,
                        bases[2].vectors, optimize='greedy').reshape(8, 8)
        assert dm2 == approx(dm)

    def test_apply_random_ptm1q_pv1q(self, state_cls, dm_basis):
        unitary = random_unitary_matrix(2, 45)

        dm_before = random_hermitian_matrix(2, seed=256)
        dm_after = unitary @ dm_before @ unitary.conj().T
        assert dm_before.trace() == approx(1)
        assert dm_after.trace() == approx(1)

        b = (dm_basis(2),)
        pv0 = state_cls.from_dm(dm_before, b)
        pv1 = state_cls.from_dm(dm_after, b)
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
    def test_apply_1q_random_ptm(self, state_cls, dm_basis, qubit):
        unity = np.identity(2)
        unitary = random_unitary_matrix(2, 45)
        einsum_args = [unity, [0, 3], unity, [1, 4], unity, [2, 5]]
        einsum_args[2*qubit] = unitary
        unitary3q = np.einsum(*einsum_args, optimize='greedy').reshape(8, 8)

        dm_before = random_hermitian_matrix(8, seed=46)
        dm_after = unitary3q @ dm_before @ unitary3q.conj().T
        assert dm_before.trace() == approx(1)
        assert dm_after.trace() == approx(1)

        b = (dm_basis(2),)
        bases = b * 3
        pv0 = state_cls.from_dm(dm_before, bases)
        pv1 = state_cls.from_dm(dm_after, bases)
        # sanity check
        for q in range(3):
            if q != qubit:
                assert pv0.meas_prob(q) == approx(pv1.meas_prob(q))
            else:
                assert pv0.meas_prob(q) != approx(pv1.meas_prob(q))

        ptm = kraus_to_ptm(unitary.reshape(1, 2, 2), b, b)
        pv0.apply_ptm(ptm, qubit)
        assert pv0.to_pv() == approx(pv1.to_pv())

    @pytest.mark.parametrize('qubit', [0, 1, 2])
    def test_apply_1q_projecting_ptm(self, state_cls, dm_basis, qubit):
        unitary = random_unitary_matrix(2, 45)

        b_in = (dm_basis(2),)
        b_out = (dm_basis(2).subbasis([1]),)
        bases = b_in * 3
        dm_before = random_hermitian_matrix(8, seed=46)
        pv = state_cls.from_dm(dm_before, bases)
        pv_before = pv.to_pv()
        ix_out = [0, 1, 2]
        ix_out[qubit] = 3
        ptm = kraus_to_ptm(unitary.reshape(1, 2, 2), b_in, b_out)

        pv_ref = np.einsum(ptm, [3, qubit], pv_before, [0, 1, 2], ix_out)
        pv.apply_ptm(ptm, qubit)
        assert pv.to_pv() == approx(pv_ref)

    @pytest.mark.parametrize('qubits', [
        (0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)
    ])
    def test_apply_2q_random_ptm(self, state_cls, dm_basis, qubits):
        unity = np.identity(2)
        unitary = random_unitary_matrix(4, 45).reshape(2, 2, 2, 2)
        decay = np.array([[0.9, 0], [0, 0.7]])
        op = decay @ unitary

        q0, q1 = qubits
        (q2,) = tuple(q for q in range(3) if q not in qubits)
        einsum_args = [op, [q0, q1, q0+3, q1+3], unity, [q2, q2+3]]
        op3q = np.einsum(*einsum_args, optimize='greedy').reshape(8, 8)

        dm_before = random_hermitian_matrix(8, seed=46)
        dm_after = op3q @ dm_before @ op3q.conj().T
        assert dm_before.trace() == approx(1)
        assert dm_after.trace() < 1.

        b = (dm_basis(2),)
        bases = b * 3
        pv0 = state_cls.from_dm(dm_before, bases)
        pv1 = state_cls.from_dm(dm_after, bases)

        ptm = kraus_to_ptm(op.reshape(1, 4, 4), b * 2, b * 2)
        pv0.apply_ptm(ptm, *qubits)
        assert pv0.to_pv() == approx(pv1.to_pv())

    @pytest.mark.parametrize(
        'bases', [
            (quantumsim.bases.general(2),
             quantumsim.bases.general(2),
             quantumsim.bases.general(2)),
            (basis_general_reshuffled,
             basis_general_reshuffled,
             basis_general_reshuffled),
            (quantumsim.bases.general(2),
             quantumsim.bases.general(2).subbasis([0, 1]),
             quantumsim.bases.general(2).subbasis([0, 1, 2]))])
    def test_diagonal_meas_prob(self, state_cls, bases):
        diag = np.array([0.25, 0, 0.75, 0, 0, 0, 0, 0])
        dm = np.diag(diag)
        s = state_cls.from_dm(dm, bases)
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
        s = state_cls.from_dm(dm, bases, qubits=['x', 'y', 'z'])
        assert s.to_pv().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)
        assert s.trace() == approx(0.75)
        assert np.allclose(s.meas_prob('x'), (0.75, 0.))
        assert np.allclose(s.meas_prob('y'), (0.75, 0.))
        assert np.allclose(s.meas_prob('z'), (0.25, 0.5))

    def test_get_diagonal(self, state_cls, dim_hilbert):
        # Default initialization
        state = state_cls(2, dim_hilbert=dim_hilbert)
        diag = state.diagonal()
        diag_ref = np.zeros(dim_hilbert**2)
        diag_ref[0] = 1.
        assert diag == approx(diag_ref)

        # Random initialization in general basis
        state = random_hermitian_matrix(dim_hilbert**2, 7654)
        diag_ref = np.diagonal(state)
        state = state_cls.from_dm(state, (quantumsim.bases.general(dim_hilbert),)*2)
        assert state.diagonal() == approx(diag_ref)

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
    def test_diagonal_indicated(self, state_cls, bases):
        dm = np.array([[min(i, j)*10 + max(i, j) for i in range(1, 9)]
                       for j in range(1, 9)])
        # dm = np.array([[min(i, j)*10 + max(i, j) for i in range(1, 5)]
        #                for j in range(1, 5)])
        diag = np.diag(dm)

        s = state_cls.from_dm(dm, bases)
        # assert s.expansion().shape == tuple(b.dim_pauli for b in bases)
        assert s.bases[0] == bases[0]
        assert s.bases[1] == bases[1]
        assert np.allclose(s.diagonal(), diag)

    def test_partial_trace(self, state_cls, dim_hilbert):
        if state_cls.__name__ == 'StateCuda':
            pytest.xfail('StateCuda.partial_trace() is not implemented')

        dm = random_hermitian_matrix(dim_hilbert**3, 826)
        bases = (quantumsim.bases.general(dim_hilbert),)*3

        # Check without reordering
        state = state_cls.from_dm(dm, bases)
        dm_traced = np.einsum('abcdec->abde', dm.reshape((dim_hilbert,)*6))\
                      .reshape(dim_hilbert**2, dim_hilbert**2)
        assert state.partial_trace(0, 1).to_dm() == approx(dm_traced)

        # Check with reordering and with symbolic qubit tags
        state = state_cls.from_dm(dm, bases, qubits=['a', 'b', 'c'])
        dm_traced = np.einsum('abcdbf->cafd', dm.reshape((dim_hilbert,)*6)) \
            .reshape(dim_hilbert**2, dim_hilbert**2)
        assert state.partial_trace('c', 'a').to_dm() == approx(dm_traced)
