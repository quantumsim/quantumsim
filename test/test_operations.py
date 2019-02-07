# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np

from qs2.operations.operation import Transformation, Chain
from qs2 import bases
from qs2.operations import qubits as lib2
from qs2.operations import qutrits as lib3
from qs2.backends import DensityMatrix


class TestOperations:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = (bases.gell_mann(2),)
        gm_two_qubit_basis = gm_qubit_basis + gm_qubit_basis

        damp_op = Transformation.from_kraus(damp_kraus_mat, 2)
        damp_ptm = damp_op.ptm(gm_qubit_basis)

        assert damp_ptm.shape == (4, 4)
        assert np.all(damp_ptm <= 1) and np.all(damp_ptm >= -1)

        expected_mat = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])

        assert np.allclose(damp_ptm, expected_mat)
        with pytest.raises(ValueError, match=r'.* should be a tuple, .*'):
            damp_op.ptm(bases.gell_mann(2), bases.gell_mann(2))

        cz_kraus_mat = np.diag([1, 1, 1, -1])
        cz_op = Transformation.from_kraus(cz_kraus_mat, 2)
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

        cz_op = Transformation.from_kraus(cz_kraus_mat, 3)
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
        kraus_op = Transformation.from_kraus(cz_kraus_mat, 2)

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            _ = Transformation.from_kraus(wrong_dim_kraus, 2)
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            _ = Transformation.from_kraus(not_sqr_kraus, 2)
        with pytest.raises(ValueError):
            _ = Transformation.from_kraus(cz_kraus_mat, 3)
        with pytest.raises(ValueError):
            _ = kraus_op.ptm(qutrit_basis+qutrit_basis)

    def test_convert_ptm_basis(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])
        gell_mann_basis = (bases.gell_mann(2),)
        general_basis = (bases.general(2),)

        damp_op_kraus = Transformation.from_kraus(damp_kraus_mat, 2)
        ptm_gell_mann = damp_op_kraus.ptm(gell_mann_basis)
        damp_op_ptm = Transformation.from_ptm(ptm_gell_mann,
                                              gell_mann_basis,
                                              gell_mann_basis)

        ptm_general_kraus = damp_op_kraus.ptm(general_basis)
        ptm_general_converted = damp_op_ptm.ptm(general_basis)

        assert np.allclose(ptm_general_kraus, ptm_general_converted)

    def test_opt_basis_single_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b1 = b.subbasis([1])
        b01 = b.computational_subbasis()

        # Identity up to floating point error
        rot = lib2.rotate_x(2 * np.pi)
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b0,))
        assert ob_in == b0
        assert ob_out == b0
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b1,))
        assert ob_in == b1
        assert ob_out == b1

        # RX(pi)
        rot = lib2.rotate_x(np.pi)
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b0,))
        assert ob_in == b0
        assert ob_out == b1
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b1,))
        assert ob_in == b1
        assert ob_out == b0

        # RY(pi/2)
        rot = lib2.rotate_y(np.pi / 2)
        (ob_in,), (ob_out,) = rot.optimal_bases(bases_in=(b01,))
        assert ob_in == b01
        assert ob_out.dim_pauli == 3
        assert '0' in ob_out.labels
        assert '1' in ob_out.labels
        assert 'X10' in ob_out.labels

    def test_opt_basis_two_qubit_2d(self):
        op = lib2.cnot()

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
        assert op_cl.shape == (1, 4, 1, 4)

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
        state1 = DensityMatrix(b)
        state2 = DensityMatrix(b)

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
