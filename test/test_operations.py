# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from numpy import pi

import qs2.operations as op
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

    def test_kraus_to_transfer_matrix(self):
        cphase_unitary = np.diag([1, 1, 1, -1])
        cphase_ptm = op.common.kraus_to_transfer_matrix(
            cphase_unitary, double_kraus=True)

        assert cphase_ptm.shape() == (16, 16)
        assert np.sum(cphase_ptm[0, :]) == 1
        assert np.sum(cphase_ptm[:, 0]) == 1
