# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from numpy import pi

from qs2.operations import common
from qs2.basis import basis
from qs2.state import State


class TestOperations:
    def test_algebra(self):
        s = State(3)
        rot_pi2 = op.rotate_y(0.5*pi)
        assert rot_pi2.n_qubits == 1

        rot_pi = rot_pi2 @ rot_pi2
        assert rot_pi.n_qubits == 1
        assert rot_pi == op.rotate_y(pi)

        rot_pi_alias = rot_pi.at(0) @ rot_pi.at(0)
        assert rot_pi == rot_pi_alias

        combined_rotation_2q = rot_pi2.at(0) @ rot_pi2.at(1)
        assert combined_rotation_2q.n_qubits == 2

        with pytest.raises(ValueError):
            # indices must be ordered
            rot_pi2.at(0) @ rot_pi2.at(2)

        combined_rotation_3q = rot_pi2[0] @ rot_pi2[1] @ rot_pi2[2]
        the_same_rotation = combined_rotation_2q.at(0, 2) @ rot_pi2.at(1)
        assert combined_rotation_3q == the_same_rotation

        combined_rotation_3q(s)
        assert rot_pi2.n_qubits == 3
        assert np.allclose(s.probability(axis='x'), [(1, 0), (1, 0), (1, 0)])

    def test_rotate_z(self):
        s = State(3)

        rotate90 = op.rotate_z(0.5*np.pi)
        rotate90(s, 1)

        rotate180 = op.rotate_z(np.pi)
        rotate180(s, 2)

        assert np.allclose(s.probability(), [(1., 0.), (0.5, 0.5), (0., 1.)])
        assert np.allclose(s.probability(0, 1), [(1., 0.), (0.5, 0.5)])
        assert np.allclose(s.probability(2), [(0., 1.)])

        rotate90(s, 0)
        assert np.allclose(s.probability(0, axis='z'), [(0.5, 0.5)])

    def test_rotate_x(self):
        s = State(3)

        rotate90 = op.rotate_x(0.5*np.pi)
        rotate90(s, 1)

        rotate180 = op.rotate_x(np.pi)
        rotate180(s, 2)

        assert np.allclose(s.probability(axis='x'),
                           [(1., 0.), (0.5, 0.5), (0., 1.)])
        assert np.allclose(s.probability(0, 1, axis='x'),
                           [(1., 0.), (0.5, 0.5)])
        assert np.allclose(s.probability(2, axis='x'), [(0., 1.)])

        rotate90(s, 0)
        assert np.allclose(s.probability(0, axis='x'), [(0.5, 0.5)])

    def test_rotate_y(self):
        raise NotImplementedError()

    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]], [[0, np.sqrt(p_damp)], [0, 0]]])
        qubit_basis = basis.gell_mann(2)
        ptm_damp = common.kraus_to_ptm(damp_kraus, qubit_basis)

        assert ptm_damp.shape == (4, 4)
        assert np.all(ptm_damp <= 1) and np.all(ptm_damp >= -1)

        expected_ptm = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])

        assert np.allclose(ptm_damp, expected_ptm)

        cz_kraus = np.diag([1, 1, 1, -1])
        system_basis = qubit_basis * qubit_basis
        cz_ptm = common.kraus_to_ptm(cz_kraus, system_basis)

        assert cz_ptm.shape == (16, 16)
        assert np.all(cz_ptm.round(3) <= 1) and np.all(cz_ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm[:, 0]), 1)

    def test_kraus_to_ptm_qutrits(self):
        cz_kraus = np.diag([1, 1, 1, 1, -1, 1, -1, 1, 1])
        qutrit_basis = basis.gell_mann(3)
        system_basis = qutrit_basis * qutrit_basis
        cz_ptm = common.kraus_to_ptm(cz_kraus, system_basis)

        assert cz_ptm.shape == (81, 81)
        assert np.all(cz_ptm.round(3) <= 1) and np.all(cz_ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm[:, 0]), 1)

    def test_kraus_to_ptm_errors(self):
        qubit_basis = basis.gell_mann(2)
        cz_kraus = np.diag([1, 1, 1, -1])

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            wrong_ptm = common.kraus_to_ptm(wrong_dim_kraus)
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            wrong_ptm = common.kraus_to_ptm(not_sqr_kraus)
        basis_mismatch_kraus = np.random.random((9, 3, 3))
        with pytest.raises(ValueError):
            wrong_ptm = common.kraus_to_ptm(
                basis_mismatch_kraus, qubit_basis)
        wrong_basis = basis.gell_mann(3)
        with pytest.raises(ValueError):
            wrong_ptm = common.kraus_to_ptm(cz_kraus, wrong_basis)
        wrong_subdims = np.array([2, 2, 2])
        with pytest.raises(ValueError):
            wrong_ptm = common.kraus_to_ptm(
                cz_kraus, subs_dim_hilbert=wrong_subdims)

    def test_ptm_to_choi(self):
        p_damp = 0.5
        qubit_basis = basis.gell_mann(2)
        ptm_damp = np.array([[1, 0, 0, 0],
                             [0, np.sqrt(1-p_damp), 0, 0],
                             [0, 0, np.sqrt(1-p_damp), 0],
                             [p_damp, 0, 0, 1-p_damp]])

        choi_damp = common.ptm_to_choi(ptm_damp, qubit_basis)

        assert choi_damp.shape == (4, 4)

        expected_choi = np.diag((1, 0, p_damp, p_damp))
        expected_choi[0, 3] = expected_choi[3, 0] = np.sqrt(p_damp)

        assert np.allclose(choi_damp, expected_choi)

    def test_ptm_to_choi_errors(self):
        not_sqr_ptm = np.random.random((4, 9))
        with pytest.raises(ValueError):
            wrong_choi = common.ptm_to_choi(not_sqr_ptm)
        wrong_dim_ptm = np.random.random((2, 4, 4))
        with pytest.raises(ValueError):
            wrong_choi = common.ptm_to_choi(wrong_dim_ptm)
        wrong_basis = basis.gell_mann(3)
        test_ptm = np.diag((1, 1, 1, 1))
        with pytest.raises(ValueError):
            wrong_choi = common.ptm_to_choi(test_ptm, wrong_basis)
        wrong_subs_dims = np.array([2, 2])
        with pytest.raises(ValueError):
            wrong_choi = common.ptm_to_choi(
                test_ptm, subs_dim_hilbert=wrong_subs_dims)

    def test_choi_to_kraus_qubits(self):
        p_damp = 0.5
        choi_damp = np.diag((1, 0, p_damp, p_damp))
        choi_damp[0, 3] = choi_damp[3, 0] = np.sqrt(p_damp)

        kraus_damp = common.choi_to_kraus(choi_damp)

        assert kraus_damp.shape == (4, 2, 2)

        expected_kraus = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]], [[0, np.sqrt(p_damp)], [0, 0]]])

        for expected_op in expected_kraus:
            assert expected_op.round(3) in kraus_damp.round(3)

    def test_choi_to_kraus_errors(self):
        not_sqr_choi = np.random.random((4, 9))
        with pytest.raises(ValueError):
            wrong_kraus = common.choi_to_kraus(not_sqr_choi)
        wrong_dim_choi = np.random.random((2, 4, 4))
        with pytest.raises(ValueError):
            wrong_kraus = common.choi_to_kraus(wrong_dim_choi)

    def test_ptm_to_kraus_qubits(self):
        p_damp = 0.5
        qubit_basis = basis.gell_mann(2)
        ptm_damp = np.array([[1, 0, 0, 0],
                             [0, np.sqrt(1-p_damp), 0, 0],
                             [0, 0, np.sqrt(1-p_damp), 0],
                             [p_damp, 0, 0, 1-p_damp]])

        kraus_damp = common.ptm_to_kraus(ptm_damp)

        assert kraus_damp.shape == (4, 2, 2)

        expected_kraus = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]], [[0, np.sqrt(p_damp)], [0, 0]]])

        for expected_op in expected_kraus:
            assert expected_op.round(3) in kraus_damp.round(3)

    def test_kraus_to_choi_qubits(self):
        p_damp = 0.5
        damp_kraus = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]], [[0, np.sqrt(p_damp)], [0, 0]]])

        choi_damp = common.kraus_to_choi(damp_kraus)

        assert choi_damp.shape == (4, 4)

        expected_choi = np.diag((1, 0, p_damp, p_damp))
        expected_choi[0, 3] = expected_choi[3, 0] = np.sqrt(p_damp)

        assert np.allclose(choi_damp, expected_choi)

    def test_kraus_to_choi_errors(self):
        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            wrong_ptm = common.kraus_to_choi(wrong_dim_kraus)
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            wrong_ptm = common.kraus_to_choi(not_sqr_kraus)
