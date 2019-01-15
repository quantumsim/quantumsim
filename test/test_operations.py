# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np

from qs2.operations import operators
from qs2 import bases
from qs2.operations import library as lib
from qs2.backends import DensityMatrix


class TestLibrary:
    def test_rotate_x(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis+qubit_basis
        dm = DensityMatrix(sys_bases)

        rotate90 = lib.rotate_x(0.5*np.pi)
        rotate90.prepare(qubit_basis)

        rotate180 = lib.rotate_x(np.pi)
        rotate180.prepare(qubit_basis)

        rotate360 = lib.rotate_x(2*np.pi)
        rotate360.prepare(qubit_basis)

        rotate90(dm, 1)
        rotate180(dm, 2)
        assert np.allclose(dm.partial_trace(0), (1, 0))
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))
        assert np.allclose(dm.partial_trace(2), (0, 1))

        rotate180(dm, 1)
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

        rotate90(dm, 1)
        assert np.allclose(dm.partial_trace(1), (1, 0))

        rotate360(dm, 0)
        assert np.allclose(dm.partial_trace(0), (1, 0))

    def test_rotate_y(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis+qubit_basis
        dm = DensityMatrix(sys_bases)

        rotate90 = lib.rotate_y(0.5*np.pi)
        rotate90.prepare(qubit_basis)

        rotate180 = lib.rotate_y(np.pi)
        rotate180.prepare(qubit_basis)

        rotate360 = lib.rotate_y(2*np.pi)
        rotate360.prepare(qubit_basis)

        rotate90(dm, 1)
        rotate180(dm, 2)
        assert np.allclose(dm.partial_trace(0), (1, 0))
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))
        assert np.allclose(dm.partial_trace(2), (0, 1))

        rotate180(dm, 1)
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

        rotate90(dm, 1)
        assert np.allclose(dm.partial_trace(1), (1, 0))

        rotate360(dm, 0)
        assert np.allclose(dm.partial_trace(0), (1, 0))

    def test_rotate_z(self):
        sqrt2 = np.sqrt(2)
        qubit_basis = (bases.general(2),)
        dm = DensityMatrix(qubit_basis)

        rotate90 = lib.rotate_z(0.5*np.pi)
        rotate90.prepare(qubit_basis)

        rotate180 = lib.rotate_z(np.pi)
        rotate180.prepare(qubit_basis)

        rotate360 = lib.rotate_z(2*np.pi)
        rotate360.prepare(qubit_basis)

        rotate90(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])
        rotate180(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])

        # manually apply a hadamard gate
        had_expansion = np.array([0.5, 0.5, sqrt2, 0])
        superpos_dm = DensityMatrix(qubit_basis,
                                    had_expansion)

        rotate180(superpos_dm, 0)
        assert np.allclose(superpos_dm.expansion(),
                           [0.5, 0.5, -sqrt2, 0])

        rotate90(superpos_dm, 0)
        assert np.allclose(superpos_dm.expansion(),
                           [0.5, 0.5, 0, -sqrt2])

        rotate180(superpos_dm, 0)
        assert np.allclose(superpos_dm.expansion(),
                           [0.5, 0.5, 0, sqrt2])

        rotate360(superpos_dm, 0)
        assert np.allclose(superpos_dm.expansion(),
                           [0.5, 0.5, 0, sqrt2])

    def test_rotate_euler(self):
        qubit_basis = (bases.general(2),)
        dm = DensityMatrix(qubit_basis+qubit_basis)

        rotate90x = lib.rotate_euler(0, 0.5*np.pi, 0)
        rotate90x.prepare(qubit_basis)

        rotate180x = lib.rotate_euler(0, np.pi, 0)
        rotate180x.prepare(qubit_basis)

        rotate90y = lib.rotate_euler(0, 0.5*np.pi, 0)
        rotate90y.prepare(qubit_basis)

        rotate180y = lib.rotate_euler(0, np.pi, 0)
        rotate180y.prepare(qubit_basis)

        rotate90x(dm, 0)
        assert np.allclose(dm.partial_trace(0), (0.5, 0.5))

        rotate90y(dm, 1)
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

    def test_hadamard(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis
        dm = DensityMatrix(sys_bases)

        hadamard = lib.hadamard()
        hadamard.prepare(qubit_basis)

        hadamard(dm, 1)
        assert np.allclose(dm.partial_trace(0), (1, 0))
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

        hadamard(dm, 1)
        assert np.allclose(dm.partial_trace(1), (1, 0))


class TestOperators:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = (bases.gell_mann(2),)
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

        cz_kraus = operators.KrausOperator(cz_kraus_mat, (2, 2))
        cz_ptm = cz_kraus.to_ptm(gm_two_qubit_basis)

        assert cz_ptm.matrix.shape == (16, 16)
        assert np.all(cz_ptm.matrix.round(3) <= 1) and np.all(
            cz_ptm.matrix.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm.matrix[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm.matrix[:, 0]), 1)

    def test_kraus_to_ptm_qutrits(self):
        cz_kraus_mat = np.diag([1, 1, 1, 1, -1, 1, -1, 1, 1])
        qutrit_basis = (bases.gell_mann(3),)
        system_bases = qutrit_basis + qutrit_basis

        cz_kraus = operators.KrausOperator(cz_kraus_mat, (3, 3))
        cz_ptm = cz_kraus.to_ptm(system_bases)

        assert cz_ptm.matrix.shape == (81, 81)
        assert np.all(cz_ptm.matrix.round(3) <= 1) and np.all(
            cz_ptm.matrix.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm.matrix[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm.matrix[:, 0]), 1)

    def test_kraus_to_ptm_errors(self):
        qutrit_basis = (bases.general(3),)
        cz_kraus_mat = np.diag([1, 1, 1, -1])
        kraus_op = operators.KrausOperator(cz_kraus_mat, (2, 2))

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            _ = operators.KrausOperator(wrong_dim_kraus, (2, 2))
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            _ = operators.KrausOperator(not_sqr_kraus, (2, 2))
        with pytest.raises(ValueError):
            _ = operators.KrausOperator(cz_kraus_mat, (3, 3))
        with pytest.raises(ValueError):
            _ = kraus_op.to_ptm(qutrit_basis+qutrit_basis)

    def test_convert_ptm_basis(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])
        gell_man_basis = (bases.gell_mann(2),)
        general_basis = (bases.general(2),)

        damp_kraus = operators.KrausOperator(damp_kraus_mat, (2,))

        ptm_gell_man = damp_kraus.to_ptm(gell_man_basis)
        ptm_general = damp_kraus.to_ptm(general_basis)

        converted_ptm = ptm_gell_man.to_ptm(general_basis)

        assert np.allclose(ptm_general.matrix, converted_ptm.matrix)
