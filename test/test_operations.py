# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np

from qs2.operations.operation import Transformation
from qs2 import bases
from qs2.operations import library as lib
from qs2.backends import DensityMatrix


class TestLibrary:
    def test_rotate_x(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis * 3
        dm = DensityMatrix(sys_bases)

        rotate90 = lib.rotate_x(0.5*np.pi)
        rotate180 = lib.rotate_x(np.pi)
        rotate360 = lib.rotate_x(2*np.pi)

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
        rotate180 = lib.rotate_y(np.pi)
        rotate360 = lib.rotate_y(2*np.pi)

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
        rotate180 = lib.rotate_z(np.pi)
        rotate360 = lib.rotate_z(2*np.pi)

        rotate90(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])
        rotate180(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])

        # manually apply a Hadamard gate
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
        rotate90y = lib.rotate_euler(0, 0.5*np.pi, 0)

        rotate90x(dm, 0)
        assert np.allclose(dm.partial_trace(0), (0.5, 0.5))

        rotate90y(dm, 1)
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

    def test_hadamard(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis
        dm = DensityMatrix(sys_bases)

        hadamard = lib.hadamard()

        hadamard(dm, 1)
        assert np.allclose(dm.partial_trace(0), (1, 0))
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

        hadamard(dm, 1)
        assert np.allclose(dm.partial_trace(1), (1, 0))


class TestOperations:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = (bases.gell_mann(2),)
        gm_two_qubit_basis = gm_qubit_basis + gm_qubit_basis

        damp_op = Transformation.from_kraus(damp_kraus_mat, [2])
        damp_ptm = damp_op.ptm(gm_qubit_basis)

        assert damp_ptm.shape == (4, 4)
        assert np.all(damp_ptm <= 1) and np.all(damp_ptm >= -1)

        expected_mat = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])

        assert np.allclose(damp_ptm, expected_mat)

        cz_kraus_mat = np.diag([1, 1, 1, -1])
        cz_op = Transformation.from_kraus(cz_kraus_mat, (2, 2))
        cz_ptm = cz_op.ptm(gm_two_qubit_basis)

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

        cz_op = Transformation.from_kraus(cz_kraus_mat, (3, 3))
        cz_ptm = cz_op.ptm(system_bases)

        assert cz_ptm.shape == (9, 9, 9, 9)
        cz_ptm_flat = cz_ptm.reshape((81, 81))
        assert np.all(cz_ptm_flat.round(3) <= 1) and np.all(
            cz_ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm_flat[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm_flat[:, 0]), 1)

    def test_kraus_to_ptm_errors(self):
        qutrit_basis = (bases.general(3),)
        cz_kraus_mat = np.diag([1, 1, 1, -1])
        kraus_op = Transformation.from_kraus(cz_kraus_mat, (2, 2))

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            _ = Transformation.from_kraus(wrong_dim_kraus, (2, 2))
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            _ = Transformation.from_kraus(not_sqr_kraus, (2, 2))
        with pytest.raises(ValueError):
            _ = Transformation.from_kraus(cz_kraus_mat, (3, 3))
        with pytest.raises(ValueError):
            _ = kraus_op.ptm(qutrit_basis+qutrit_basis)

    def test_convert_ptm_basis(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])
        gell_man_basis = (bases.gell_mann(2),)
        general_basis = (bases.general(2),)

        damp_op_kraus = Transformation.from_kraus(damp_kraus_mat, (2,))
        ptm_gell_man = damp_op_kraus.ptm(gell_man_basis)
        damp_op_ptm = Transformation.from_ptm(ptm_gell_man,
                                              gell_man_basis,
                                              gell_man_basis)

        ptm_general_kraus = damp_op_kraus.ptm(general_basis)
        ptm_general_converted = damp_op_ptm.ptm(general_basis)

        assert np.allclose(ptm_general_kraus, ptm_general_converted)

    def test_opt_basis_singleqb_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b1 = b.subbasis([1])
        b01 = b.computational_subbasis()

        # Identity up to floating point error
        rot = lib.rotate_x(2*np.pi)
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b0,))
        assert ob_in == b0
        assert ob_out == b0
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b1,))
        assert ob_in == b1
        assert ob_out == b1

        # RX(pi)
        rot = lib.rotate_x(np.pi)
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b0,))
        assert ob_in == b0
        assert ob_out == b1
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b1,))
        assert ob_in == b1
        assert ob_out == b0

        # RY(pi/2)
        rot = lib.rotate_y(np.pi/2)
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b01,))
        assert ob_in == b01
        assert ob_out.dim_pauli == 3
        assert '0' in ob_out.labels
        assert '1' in ob_out.labels
        assert 'X10' in ob_out.labels

    def test_opt_basis_twoqb_2d(self):
        op = lib.cnot()

        # Classical input basis -> classical output basis
        # Possible flip in control bit
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.subbasis([0, 1])
        b_in = (b01, b0)
        ob_in, ob_out = op.optimal_bases(bases_in=b_in)
        assert ob_in[0] == b01
        assert ob_in[1] == b0
        assert ob_out[0] == b01
        assert ob_out[1] == b01

        # Classical control bit is not violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b0, b)
        ob_in, ob_out = op.optimal_bases(bases_in=b_in)
        assert ob_in[0] == b0
        assert ob_in[1] == b
        assert ob_out[0] == b0
        assert ob_out[1] == b

        # Classical target bit will become quantum for quantum control bit,
        # input should not be violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b, b0)
        ob_in, ob_out = op.optimal_bases(bases_in=b_in)
        assert ob_in[0] == b
        assert ob_in[1] == b0
        assert ob_out[0] == b
        assert ob_out[1] == b

    def test_compile_singleqb_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.computational_subbasis()

        op = lib.rotate_y(np.pi)
        assert op.shape == (4, 4)
        op_full = op.compile(bases_in=(b,))
        assert op_full.shape == (4, 4)
        op_cl = op.compile(bases_in=(b01,))
        assert op_cl.shape == (2, 2)

        op = lib.rotate_x(np.pi/3)
        assert op.shape == (4, 4)
        op_full = op.compile(bases_in=(b,), bases_out=(b01,))
        # X component of a state is irrelevant for the output.
        assert op_full.shape == (2, 3)
        op_cl = op.compile(bases_in=(b0,))
        assert op_cl.shape == (3, 1)

    def test_compile_twoqb_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b1 = b.subbasis([1])
        b01 = b.computational_subbasis()

        op = lib.cnot()
        assert op.shape == (4, 4, 4, 4)
        op_full = op.compile(bases_in=(b, b))
        assert op_full.shape == (4, 4, 4, 4)
        op_cl = op.compile(bases_in=(b01, b01))
        assert op_cl.shape == (2, 2, 2, 2)
        op_cl = op.compile(bases_in=(b0, b))
        assert op_cl.shape == (1, 4, 1, 4)
