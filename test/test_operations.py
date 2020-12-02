# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from pytest import approx

from quantumsim.models import perfect_qubits as lib2
from quantumsim import bases, State
from quantumsim.algebra.tools import random_hermitian_matrix
from quantumsim.circuits import Gate


# noinspection DuplicatedCode
class TestOperations:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = (bases.gell_mann(2),)
        gm_two_qubit_basis = gm_qubit_basis * 2

        damp_op = Gate.from_kraus(damp_kraus_mat, gm_qubit_basis)

        expected_mat = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])
        assert np.allclose(damp_op.ptm, expected_mat)

        with pytest.raises(ValueError, match=r'.* must be list-like, .*'):
            damp_op.set_bases(bases.gell_mann(2), bases.gell_mann(2))

        cz_kraus_mat = np.diag([1, 1, 1, -1])
        cz = Gate.from_kraus(cz_kraus_mat, gm_two_qubit_basis)
        cz_ptm = cz.ptm

        assert cz_ptm.shape == (4, 4, 4, 4)
        cz_ptm = cz_ptm.reshape((16, 16))
        assert np.all(cz_ptm.round(3) <= 1)
        assert np.all(cz_ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm[:, 0]), 1)

    def test_kraus_to_ptm_qutrits(self):
        cz_kraus_mat = np.diag([1, 1, 1, 1, -1, 1, -1, 1, 1])
        qutrit_basis = (bases.gell_mann(3),)
        system_bases = qutrit_basis * 2

        cz = Gate.from_kraus(cz_kraus_mat, system_bases)
        cz_ptm = cz.ptm

        assert cz_ptm.shape == (9, 9, 9, 9)
        cz_ptm_flat = cz_ptm.reshape((81, 81))
        assert np.all(cz_ptm_flat.round(3) <= 1) and np.all(
            cz_ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm_flat[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm_flat[:, 0]), 1)

    def test_kraus_to_ptm_errors(self):
        qubit_basis = (bases.general(2),)
        qutrit_basis = (bases.general(3),)
        cz_kraus_mat = np.diag([1, 1, 1, -1])
        kraus_op = Gate.from_kraus(cz_kraus_mat, qubit_basis*2)

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            _ = Gate.from_kraus(wrong_dim_kraus, qubit_basis)
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            _ = Gate.from_kraus(not_sqr_kraus, qubit_basis)
        with pytest.raises(ValueError):
            _ = Gate.from_kraus(cz_kraus_mat, qutrit_basis)
        with pytest.raises(ValueError):
            _ = kraus_op.set_bases(qutrit_basis*2)

    def test_convert_ptm_basis(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])
        gell_mann_basis = (bases.gell_mann(2),)
        general_basis = (bases.general(2),)

        op1 = Gate.from_kraus(damp_kraus_mat, gell_mann_basis)
        op2 = op1.set_bases(general_basis, general_basis) \
            .set_bases(gell_mann_basis, gell_mann_basis)
        assert op1.ptm == approx(op2.ptm)
        assert op1.bases_in == op2.bases_in
        assert op1.bases_out == op2.bases_out

    def test_lindblad_single_qubit(self):
        ham = random_hermitian_matrix(2, seed=56)
        lindblad_ops = np.array([
            [[0, 0.1],
             [0, 0]],
            [[0, 0],
             [0, 0.33]],
        ])
        t1 = 10
        t2 = 25
        b1 = (bases.general(2),)
        b2 = (bases.gell_mann(2),)
        op1 = Gate.from_lindblad_form(t1, b1, b2, hamiltonian=ham,
                                      lindblad_ops=lindblad_ops)
        op2 = Gate.from_lindblad_form(t2, b2, b1, hamiltonian=ham,
                                      lindblad_ops=lindblad_ops)
        op = Gate.from_lindblad_form(t1+t2, b1, hamiltonian=ham,
                                     lindblad_ops=lindblad_ops)
        dm = random_hermitian_matrix(2, seed=3)
        state1 = State.from_dm(dm, b1)
        state2 = State.from_dm(dm, b1)

        op1 @ state1
        op2 @ state1
        op @ state2
        assert np.allclose(state1.to_pv(), state2.to_pv())

    def test_lindblad_time_inverse(self):
        ham = random_hermitian_matrix(2, seed=4)
        b = (bases.general(2),)
        op_plus = Gate.from_lindblad_form(20, b, hamiltonian=ham)
        op_minus = Gate.from_lindblad_form(20, b, hamiltonian=-ham)
        dm = random_hermitian_matrix(2, seed=5)
        state = State.from_dm(dm, b)
        op_plus @ state
        op_minus @ state
        assert np.allclose(state.to_dm(), dm)

    def test_lindblad_two_qubit(self):
        b = (bases.general(2),)
        identity = np.array([[1, 0], [0, 1]])
        ham1 = random_hermitian_matrix(2, seed=6)
        ham2 = random_hermitian_matrix(2, seed=7)
        ham = (np.kron(ham1, identity).reshape((2, 2, 2, 2)) +
               np.kron(identity, ham2).reshape((2, 2, 2, 2)))
        dm = random_hermitian_matrix(4, seed=3)
        op1 = Gate.from_lindblad_form(25, b, hamiltonian=ham1)
        op2 = Gate.from_lindblad_form(25, b, hamiltonian=ham2, qubits=1)
        op = Gate.from_lindblad_form(25, b*2, hamiltonian=ham)
        state1 = State.from_dm(dm, b*2)
        state2 = State.from_dm(dm, b*2)
        op1 @ state1
        op2 @ state1
        op @ state2
        assert np.allclose(state1.to_pv(), state2.to_pv())

        ops1 = np.array([
            [[0, 0.1],
             [0, 0]],
            [[0, 0],
             [0, 0.33]],
        ])
        ops2 = np.array([
            [[0, 0.15],
             [0, 0]],
            [[0, 0],
             [0, 0.17]],
        ])
        ops = [np.kron(op, identity).reshape((2, 2, 2, 2)) for op in ops1] + \
              [np.kron(identity, op).reshape((2, 2, 2, 2)) for op in ops2]
        op1 = Gate.from_lindblad_form(25, b, lindblad_ops=ops1)
        op2 = Gate.from_lindblad_form(25, b, lindblad_ops=ops2, qubits=1)
        op = Gate.from_lindblad_form(25, b*2, lindblad_ops=ops)
        state1 = State.from_dm(dm, b*2)
        state2 = State.from_dm(dm, b*2)
        op1 @ state1
        op2 @ state1
        op @ state2
        assert np.allclose(state1.to_pv(), state2.to_pv())

        op1 = Gate.from_lindblad_form(25, b, hamiltonian=ham1,
                                      lindblad_ops=ops1)
        op2 = Gate.from_lindblad_form(25, b, hamiltonian=ham2,
                                      lindblad_ops=ops2, qubits=1)
        op = Gate.from_lindblad_form(25, b*2, hamiltonian=ham,
                                     lindblad_ops=ops)
        state1 = State.from_dm(dm, b*2)
        state2 = State.from_dm(dm, b*2)
        op1 @ state1
        op2 @ state1
        op @ state2
        assert np.allclose(state1.to_pv(), state2.to_pv())

    def test_ptm(self):
        # Some random gate sequence
        circuit = (lib2.rotate_x(0, angle=np.pi/2) +
                   lib2.rotate_y(1, angle=0.3333) +
                   lib2.cphase(0, 2, angle=np.pi) +
                   lib2.cphase(1, 2, angle=np.pi) +
                   lib2.rotate_x(1, angle=-np.pi/2))

        b = (bases.general(2),) * 3
        ptm = circuit.finalize().ptm(b, b)
        assert isinstance(ptm, np.ndarray)

        op_3q = Gate.from_ptm(ptm, b)
        dm = random_hermitian_matrix(8, seed=93)
        state1 = State.from_dm(dm, b)
        state2 = State.from_dm(dm, b)

        circuit @ state1
        op_3q @ state2
        assert np.allclose(state1.to_pv(), state2.to_pv())
