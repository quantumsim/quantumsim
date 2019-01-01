# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from numpy import pi

import qs2.operations as op
from qs2.operations import operators
from qs2 import bases
from qs2.state import State


class TestOperations:
    @pytest.mark.skip(reason='Not implemented yet')
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

        combined_rotation_3q = rot_pi2[0] @ rot_pi2[1] @ rot_pi2[2]
        the_same_rotation = combined_rotation_2q.at(0, 2) @ rot_pi2.at(1)
        assert combined_rotation_3q == the_same_rotation

        combined_rotation_3q(s)
        assert rot_pi2.n_qubits == 3
        assert np.allclose(s.probability(axis='x'), [(1, 0), (1, 0), (1, 0)])

    @pytest.mark.skip(reason='Not implemented yet')
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

    @pytest.mark.skip(reason='Not implemented yet')
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

    @pytest.mark.skip(reason='Not implemented yet')
    def test_rotate_y(self):
        raise NotImplementedError()


class TestOperators:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = [bases.gell_mann(2)]
        gm_two_qubit_basis = gm_qubit_basis + gm_qubit_basis

        damp_kraus = operators.KrausOperator(damp_kraus_mat, [2])

        damp_ptm = damp_kraus.to_ptm(gm_qubit_basis)

        assert damp_ptm.matrix.shape == (4, 4)
        assert np.all(damp_ptm.matrix <= 1) and np.all(damp_ptm.matrix >= -1)

        expected_mat = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])

        assert np.allclose(damp_ptm.matrix, expected_mat)

        cz_kraus_mat = np.diag([1, 1, 1, -1])

        cz_kraus = operators.KrausOperator(cz_kraus_mat, [2, 2])
        cz_ptm = cz_kraus.to_ptm(gm_two_qubit_basis)

        assert cz_ptm.matrix.shape == (16, 16)
        assert np.all(cz_ptm.matrix.round(3) <= 1) and np.all(
            cz_ptm.matrix.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm.matrix[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm.matrix[:, 0]), 1)

    def test_kraus_to_ptm_qutrits(self):
        cz_kraus_mat = np.diag([1, 1, 1, 1, -1, 1, -1, 1, 1])
        qutrit_basis = [bases.gell_mann(3)]
        system_bases = qutrit_basis + qutrit_basis

        cz_kraus = operators.KrausOperator(cz_kraus_mat, [3, 3])
        cz_ptm = cz_kraus.to_ptm(system_bases)

        assert cz_ptm.matrix.shape == (81, 81)
        assert np.all(cz_ptm.matrix.round(3) <= 1) and np.all(
            cz_ptm.matrix.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm.matrix[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm.matrix[:, 0]), 1)

    def test_kraus_to_ptm_errors(self):
        qubit_basis = [bases.general(2)]
        qutrit_basis = [bases.general(3)]
        cz_kraus_mat = np.diag([1, 1, 1, -1])
        kraus_op = operators.KrausOperator(cz_kraus_mat, [2, 2])

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            _ = operators.KrausOperator(wrong_dim_kraus, [2, 2])
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            _ = operators.KrausOperator(not_sqr_kraus, [2, 2])
        with pytest.raises(ValueError):
            _ = operators.KrausOperator(cz_kraus_mat, [3, 3])
        with pytest.raises(ValueError):
            _ = kraus_op.to_ptm(qutrit_basis+qutrit_basis)

    def test_convert_ptm_basis(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])
        gell_man_basis = [bases.gell_mann(2)]
        general_basis = [bases.general(2)]

        damp_kraus = operators.KrausOperator(damp_kraus_mat, [2])

        ptm_gell_man = damp_kraus.to_ptm(gell_man_basis)
        ptm_general = damp_kraus.to_ptm(general_basis)

        converted_ptm = ptm_gell_man.to_ptm(general_basis)

        assert np.allclose(ptm_general.matrix, converted_ptm.matrix)
