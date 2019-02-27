# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from pytest import approx
from scipy.stats import unitary_group

from qs2.operations.operation import Operation, Chain
from qs2 import bases
from qs2.models import qubits as lib2
from qs2.models import transmons as lib3
from qs2.states import State


def random_hermitean_matrix(dim, seed):
    rng = np.random.RandomState(seed)
    dm = rng.randn(dim, dim) + 1j * rng.randn(dim, dim)
    dm += dm.conjugate().transpose()
    dm /= dm.trace()
    return dm


def random_unitary_matrix(dim, seed):
    rng = np.random.RandomState(seed)
    return unitary_group.rvs(dim, random_state=rng)


@pytest.fixture(params=['numpy', 'cuda'])
def dm_class(request):
    mod = pytest.importorskip('qs2.states.' + request.param)
    return mod.DensityMatrix


class TestOperations:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = (bases.gell_mann(2),)
        gm_two_qubit_basis = gm_qubit_basis + gm_qubit_basis

        damp_op = Operation.from_kraus(damp_kraus_mat, 2)
        damp_ptm = damp_op.compile(bases_in=gm_qubit_basis,
                                   bases_out=gm_qubit_basis).ptm

        expected_mat = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])
        assert np.allclose(damp_ptm, expected_mat)

        with pytest.raises(ValueError, match=r'.* should be a tuple, .*'):
            damp_op.compile(bases.gell_mann(2), bases.gell_mann(2))

        cz_kraus_mat = np.diag([1, 1, 1, -1])
        cz = Operation.from_kraus(cz_kraus_mat, 2).compile(
            gm_two_qubit_basis, gm_two_qubit_basis)

        assert cz.ptm.shape == (4, 4, 4, 4)
        cz_ptm = cz.ptm.reshape((16, 16))
        assert np.all(cz_ptm.round(3) <= 1)
        assert np.all(cz_ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm[:, 0]), 1)

    def test_kraus_to_ptm_qutrits(self):
        cz_kraus_mat = np.diag([1, 1, 1, 1, -1, 1, -1, 1, 1])
        qutrit_basis = (bases.gell_mann(3),)
        system_bases = qutrit_basis * 2

        cz = Operation.from_kraus(cz_kraus_mat, 3).compile(
            system_bases, system_bases)

        assert cz.ptm.shape == (9, 9, 9, 9)
        cz_ptm_flat = cz.ptm.reshape((81, 81))
        assert np.all(cz_ptm_flat.round(3) <= 1) and np.all(
            cz.ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm_flat[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm_flat[:, 0]), 1)

    def test_kraus_to_ptm_errors(self):
        qutrit_basis = (bases.general(3),)
        cz_kraus_mat = np.diag([1, 1, 1, -1])
        kraus_op = Operation.from_kraus(cz_kraus_mat, 2)

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            _ = Operation.from_kraus(wrong_dim_kraus, 2)
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            _ = Operation.from_kraus(not_sqr_kraus, 2)
        with pytest.raises(ValueError):
            _ = Operation.from_kraus(cz_kraus_mat, 3)
        with pytest.raises(ValueError):
            _ = kraus_op.compile(qutrit_basis+qutrit_basis)

    def test_convert_ptm_basis(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])
        gell_mann_basis = (bases.gell_mann(2),)
        general_basis = (bases.general(2),)

        damp_op_kraus = Operation.from_kraus(damp_kraus_mat, 2)
        op1 = damp_op_kraus.compile(gell_mann_basis, gell_mann_basis)
        op2 = damp_op_kraus.compile(general_basis, general_basis) \
                           .compile(gell_mann_basis, gell_mann_basis)
        assert np.allclose(op1.ptm, op2.ptm)
        assert op1.bases_in == op2.bases_in
        assert op1.bases_out == op2.bases_out


    def test_opt_basis_single_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b1 = b.subbasis([1])
        b01 = b.computational_subbasis()

        # Identity up to floating point error
        rot = lib2.rotate_x(2 * np.pi).compile(bases_in=(b0,), optimize=True)
        assert rot.bases_in == (b0,)
        assert rot.bases_out == (b0,)
        rot = lib2.rotate_x(2 * np.pi).compile(bases_in=(b1,), optimize=True)
        assert rot.bases_in == (b1,)
        assert rot.bases_out == (b1,)

        # RX(pi)
        rot = lib2.rotate_x(np.pi).compile(bases_in=(b0,))
        assert rot.bases_in == (b0,)
        assert rot.bases_out == (b1,)
        rot = lib2.rotate_x(np.pi).compile(bases_in=(b1,))
        assert rot.bases_in == (b1,)
        assert rot.bases_out == (b0,)

        # RY(pi/2)
        rot = lib2.rotate_y(np.pi / 2).compile(bases_in=(b01,))
        assert rot.bases_in[0] == b01
        assert rot.bases_out[0].dim_pauli == 3
        assert '0' in rot.bases_out[0].labels
        assert '1' in rot.bases_out[0].labels
        assert 'X10' in rot.bases_out[0].labels

    def test_opt_basis_two_qubit_2d(self):
        op = lib2.cnot()

        # Classical input basis -> classical output basis
        # Possible flip in control bit
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.subbasis([0, 1])
        b_in = (b01, b0)
        op_c = op.compile(bases_in=b_in, optimize=True)
        assert op_c.bases_in[0] == b01
        assert op_c.bases_in[1] == b0
        assert op_c.bases_out[0] == b01
        assert op_c.bases_out[1] == b01

        # Classical control bit is not violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b0, b)
        op_c = op.compile(bases_in=b_in, optimize=True)
        assert op_c.bases_in[0] == b0
        assert op_c.bases_in[1] == b
        assert op_c.bases_out[0] == b0
        assert op_c.bases_out[1] == b

        # Classical target bit will become quantum for quantum control bit,
        # input should not be violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b, b0)
        op_c = op.compile(bases_in=b_in, optimize=True)
        assert op_c.bases_in[0] == b
        assert op_c.bases_in[1] == b0
        assert op_c.bases_out[0] == b
        assert op_c.bases_out[1] == b

    def test_compile_single_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.computational_subbasis()

        op = lib2.rotate_y(np.pi)
        assert op.shape == (4, 4)
        op_full = op.compile(bases_in=(b,))
        assert op_full.shape == (4, 4)
        op_cl = op.compile(bases_in=(b01,))
        assert op_cl.shape == (2, 2)

        op = lib2.rotate_x(np.pi / 3)
        assert op.shape == (4, 4)
        op_full = op.compile(bases_in=(b,), bases_out=(b01,))
        # X component of a state is irrelevant for the output.
        assert op_full.shape == (2, 3)
        op_cl = op.compile(bases_in=(b0,))
        assert op_cl.shape == (3, 1)

    def test_compile_two_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.computational_subbasis()

        op = lib2.cnot()
        assert op.shape == (4, 4, 4, 4)
        op_full = op.compile(bases_in=(b, b))
        assert op_full.shape == (4, 4, 4, 4)
        op_cl = op.compile(bases_in=(b01, b01))
        assert op_cl.shape == (2, 2, 2, 2)
        op_cl = op.compile(bases_in=(b0, b))
        assert op_cl.shape == (4, 1, 4, 1)

    def test_chain_create(self):
        op1 = lib2.rotate_x()
        op2 = lib2.rotate_y()
        op3 = lib2.cnot()
        op_qutrit = lib3.rotate_x()

        circuit = Chain(op1.at(0), op2.at(0))
        assert circuit.num_qubits == 1
        assert len(circuit.operations) == 2

        circuit = Chain(op1.at(1), op2.at(0))
        assert circuit.num_qubits == 2
        assert len(circuit.operations) == 2

        with pytest.raises(ValueError, match=".* must form an ordered set .*"):
            Chain(op1.at(2), op2.at(0))

        with pytest.raises(ValueError, match=".* must form an ordered set .*"):
            Chain(op1.at(1), op2.at(2))

        with pytest.raises(ValueError, match=".* the same Hilbert "
                                             "dimensionality.*"):
            Chain(op1.at(0), op_qutrit.at(0))

        circuit3q = Chain(op1.at(0), op2.at(1), op3.at(0, 1),
                          op1.at(1), op2.at(0), op3.at(0, 2))
        assert circuit3q.num_qubits == 3
        assert len(circuit3q.operations) == 6

        with pytest.raises(ValueError, match="Number of indices is not .*"):
            Chain(op1.at(0), op3.at(0))

        with pytest.raises(ValueError, match="Number of indices is not .*"):
            circuit3q.at(0, 1)

        circuit4q = Chain(op3.at(0, 2), circuit3q.at(1, 2, 3))
        assert len(circuit4q.operations) == 7
        assert circuit4q.operations[0].indices == (0, 2)
        for o1, o2 in zip(circuit4q.operations[1:], circuit3q.operations):
            assert np.all(np.array(o1.indices) == np.array(o2.indices) + 1)

        circuit4q = Chain(circuit3q.at(2, 0, 3), op3.at(0, 1), op2.at(1))
        assert len(circuit4q.operations) == 8
        assert circuit4q.operations[0].indices == (2,)
        assert circuit4q.operations[1].indices == (0,)
        assert circuit4q.operations[2].indices == (2, 0)

    def test_chain_apply(self):
        b = (bases.general(2),) * 3
        state1 = State(b)
        state2 = State(b)

        # Some random gate sequence
        op_indices = [(lib2.rotate_x(np.pi/2), (0,)),
                      (lib2.rotate_y(0.3333), (1,)),
                      (lib2.cphase(), (0, 2)),
                      (lib2.cphase(), (1, 2)),
                      (lib2.rotate_x(-np.pi/2), (0,))]

        for op, indices in op_indices:
            op(state1, *indices),

        circuit = Chain(*(op.at(*ix) for op, ix in op_indices))
        circuit(state2, 0, 1, 2)

        assert np.all(state1.expansion() == state2.expansion())

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_compile_single_qubit(self, d, lib):
        b = bases.general(d)

        bases_full = (b,)
        subbases = (b.subbasis([0, 1]),)
        angle = np.pi/5
        rx_angle = lib.rotate_x(angle)
        rx_2angle = lib.rotate_x(2*angle)
        chain0 = Chain(rx_angle.at(0), rx_angle.at(0))
        assert chain0.num_qubits == 1
        assert len(chain0.operations) == 2

        chain1 = chain0.compile(bases_full, bases_full)
        assert chain1.num_qubits == 1
        assert len(chain1.operations) == 1
        op_angle = chain1.operations[0].operation
        op_2angle = rx_2angle.compile(bases_full, bases_full)
        assert op_angle.shape == op_2angle.shape
        assert op_angle.bases_in == op_2angle.bases_in
        assert op_angle.bases_out == op_2angle.bases_out
        assert op_angle.ptm == approx(op_2angle.ptm)

        rx_pi = lib.rotate_x(np.pi)
        chain_2pi = Chain(rx_pi.at(0), rx_pi.at(0))
        chain2 = chain_2pi.compile(subbases, bases_full)
        op = chain2.operations[0].operation
        assert op.bases_in == subbases
        assert op.bases_out == subbases

        chain3 = chain_2pi.compile(bases_full, subbases)
        op = chain3.operations[0].operation
        assert op.bases_in == subbases
        assert op.bases_out == subbases

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_merge_next(self, dm_class, d, lib):
        b = bases.general(d)

        dm = random_hermitean_matrix(seed=42)

        chain = Chain(
            lib.rotate_x(np.pi / 5).at(0),
            (lib.cphase(angle=3*np.pi/7, leakage=0.1)
             if d == 3 else lib.cphase(3*np.pi / 7)).at(1, 0),
        )

        bases_full = (b, b)
        chain_c = chain.compile(bases_full, bases_full)
        assert len(chain.operations) == 2
        assert len(chain_c.operations) == 1

        state1 = dm_class.from_dm(bases_full, dm)
        state2 = dm_class.from_dm(bases_full, dm)
        chain(state1, 0, 1)
        chain_c(state2, 0, 1)

        assert np.allclose(state1.meas_prob(0), state2.meas_prob(0))
        assert np.allclose(state1.meas_prob(1), state2.meas_prob(1))

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_merge_prev(self, d, lib):
        b = bases.general(d)

        rng = np.random.RandomState(4242)
        dm = rng.randn(d*d, d*d) + 1j * rng.randn(d*d, d*d)
        dm += dm.conjugate().transpose()
        dm /= dm.trace()

        chain = Chain(
            (lib.cphase(angle=np.pi/7, leakage=0.25)
             if d == 3 else lib.cphase(3*np.pi / 7)).at(0, 1),
            lib.rotate_x(4 * np.pi / 7).at(0),
        )

        bases_full = (b, b)
        chain_c = chain.compile(bases_full, bases_full)
        assert len(chain.operations) == 2
        assert len(chain_c.operations) == 1

        state1 = State.from_dm(bases_full, dm)
        state2 = State.from_dm(bases_full, dm)
        chain(state1, 0, 1)
        chain_c(state2, 0, 1)

        assert np.allclose(state1.meas_prob(0), state2.meas_prob(0))
        assert np.allclose(state1.meas_prob(1), state2.meas_prob(1))

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_compile_three_qubit(self, d, lib):
        b = bases.general(d)
        b0 = b.subbasis([0])

        chain0 = Chain(
            lib.rotate_x(0.5*np.pi).at(2),
            lib.cphase().at(0, 2),
            lib.cphase().at(1, 2),
            lib.rotate_x(-0.75*np.pi).at(2),
            lib.rotate_x(0.25*np.pi).at(2),
        )
        chain1 = chain0.compile((b, b, b0), (b, b, b))
        assert chain1.operations[0].indices == (0, 2)
        assert chain1.operations[0].operation.bases_in == (b, b0)
        assert chain1.operations[0].operation.bases_out[0] == b
        assert chain1.operations[1].indices == (1, 2)
        assert chain1.operations[1].operation.bases_in[0] == b
        assert chain1.operations[1].operation.bases_out[0] == b
        for label in '0', '1', 'X10', 'Y10':
            assert label in chain1.operations[1].operation.bases_out[1].labels

    def test_chain_compile_leaking(self):
        b = bases.general(3)
        chain0 = Chain(
            lib3.rotate_x(0.5*np.pi).at(2),
            lib3.cphase(leakage=0.1).at(0, 2),
            lib3.cphase(leakage=0.1).at(1, 2),
            lib3.rotate_x(-0.75*np.pi).at(2),
            lib3.rotate_x(0.25*np.pi).at(2),
        )
        b0 = b.subbasis([0])
        b01 = b.subbasis([0, 1])
        b0134 = b.subbasis([0, 1, 3, 4])
        chain1 = chain0.compile((b0, b0, b0134), (b, b, b))
        # Ancilla is not leaking here
        anc_basis = chain1.operations[1].operation.bases_out[1]
        for label in anc_basis.labels:
            assert '2' not in label

        chain2 = chain0.compile((b01, b01, b0134), (b, b, b))
        # Ancilla is leaking here
        anc_basis = chain2.operations[1].operation.bases_out[1]
        for label in '2', 'X20', 'Y20', 'X21', 'Y21':
            assert label in anc_basis.labels

    def test_zz_parity_compilation(self, dm_class):
        b_full = bases.general(3)
        b0 = b_full.subbasis([0])
        b01 = b_full.subbasis([0, 1])
        b012 = b_full.subbasis([0, 1, 2])

        bases_in = (b01, b01, b0)
        bases_out = (b_full, b_full, b012)
        zz = Chain(
            lib3.rotate_x(-np.pi/2).at(2),
            lib3.cphase(leakage=0.1).at(0, 2),
            # lib3.cphase(leakage=0.25).at(2, 1),
            # lib3.rotate_x(np.pi/2).at(2),
            # lib3.rotate_x(np.pi).at(0),
            lib3.rotate_x(np.pi).at(1)
        )
        zzc = zz.compile(bases_in=bases_in, bases_out=bases_out)
        # assert np.allclose(state1.expansion(), state2.expansion())

        # assert len(zzc.operations) == 2
        # op1, ix1 = zzc.operations[0]
        # op2, ix2 = zzc.operations[1]
        # assert ix1 == (0, 2)
        # assert ix2 == (2, 1)
        # assert op1.bases_in == (
        #     b_full.subbasis([0, 1]),
        #     b_full.subbasis([0]),
        # )
        # assert op1.bases_out == (
        #     b_full.subbasis([0, 1, 3, 4]),
        #     b_full.subbasis([0, 1, 2, 3, 5, 6, 7, 8]),
        # )
        # assert op2.bases_in == (
        #     b_full.subbasis([0, 1, 2, 3, 5, 6, 7, 8]),
        #     b_full.subbasis([0, 1]),
        # )
        # assert op1.bases_out == (
        #     b_full.subbasis([0, 1, 2]),
        #     b_full.subbasis([0, 1, 2, 3, 5, 6, 7, 8]),
        # )
        diag = np.zeros(27, dtype=complex)
        diag[0] = 0.5
        diag[4] = 0.5
        dm = np.diag(diag)
        state1 = dm_class.from_dm((b01, b01, b0), dm)
        state2 = dm_class.from_dm((b01, b01, b0), dm)

        zz(state1, 0, 1, 2)
        zzc(state2, 0, 1, 2)

        # assert np.allclose(state1.diagonal(), state2.diagonal())
        # assert np.allclose(state1.meas_prob(0), state2.meas_prob(0))
        assert state1.meas_prob(1) == approx(state2.meas_prob(1))
        # assert np.allclose(state1.meas_prob(2), state2.meas_prob(2))
