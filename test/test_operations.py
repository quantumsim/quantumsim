# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np

from quantumsim import bases, Operation
from quantumsim.algebra.tools import random_density_matrix
from quantumsim.pauli_vectors import PauliVectorNumpy as PauliVector
from quantumsim.models import qubits as lib2
from quantumsim.models import transmons as lib3


class TestOperations:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = (bases.gell_mann(2),)
        gm_two_qubit_basis = gm_qubit_basis * 2

        damp_op = Operation.from_kraus(damp_kraus_mat, gm_qubit_basis)
        damp_ptm = damp_op.ptm

        expected_mat = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])
        assert np.allclose(damp_ptm, expected_mat)

        with pytest.raises(ValueError, match=r'.* must be list-like, .*'):
            damp_op.set_bases(bases.gell_mann(2), bases.gell_mann(2))

        cz_kraus_mat = np.diag([1, 1, 1, -1])
        cz = Operation.from_kraus(cz_kraus_mat, gm_two_qubit_basis)

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

        cz = Operation.from_kraus(cz_kraus_mat, system_bases)

        assert cz.ptm.shape == (9, 9, 9, 9)
        cz_ptm_flat = cz.ptm.reshape((81, 81))
        assert np.all(cz_ptm_flat.round(3) <= 1) and np.all(
            cz.ptm.round(3) >= -1)
        assert np.isclose(np.sum(cz_ptm_flat[0, :]), 1)
        assert np.isclose(np.sum(cz_ptm_flat[:, 0]), 1)

    def test_kraus_to_ptm_errors(self):
        qubit_basis = (bases.general(2),)
        qutrit_basis = (bases.general(3),)
        cz_kraus_mat = np.diag([1, 1, 1, -1])
        kraus_op = Operation.from_kraus(cz_kraus_mat, qubit_basis*2)

        wrong_dim_kraus = np.random.random((4, 4, 2, 2))
        with pytest.raises(ValueError):
            _ = Operation.from_kraus(wrong_dim_kraus, qubit_basis)
        not_sqr_kraus = np.random.random((4, 2, 3))
        with pytest.raises(ValueError):
            _ = Operation.from_kraus(not_sqr_kraus, qubit_basis)
        with pytest.raises(ValueError):
            _ = Operation.from_kraus(cz_kraus_mat, qutrit_basis)
        with pytest.raises(ValueError):
            _ = kraus_op.set_bases(qutrit_basis*2)

    def test_convert_ptm_basis(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1-p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])
        gell_mann_basis = (bases.gell_mann(2),)
        general_basis = (bases.general(2),)

        op1 = Operation.from_kraus(damp_kraus_mat, gell_mann_basis)
        op2 = op1.set_bases(general_basis, general_basis) \
            .set_bases(gell_mann_basis, gell_mann_basis)
        assert np.allclose(op1.ptm, op2.ptm)
        assert op1.bases_in == op2.bases_in
        assert op1.bases_out == op2.bases_out

    def test_chain_create(self):
        op1 = lib2.rotate_x()
        op2 = lib2.rotate_y()
        op3 = lib2.cnot()
        op_qutrit = lib3.rotate_x()

        circuit = Operation.from_sequence(op1.at(0), op2.at(0))
        assert circuit.num_qubits == 1
        assert len(circuit.operations) == 2

        circuit = Operation.from_sequence(op1.at(1), op2.at(0))
        assert circuit.num_qubits == 2
        assert len(circuit.operations) == 2

        with pytest.raises(ValueError, match=".* must form an ordered set .*"):
            Operation.from_sequence(op1.at(2), op2.at(0))

        with pytest.raises(ValueError, match=".* must form an ordered set .*"):
            Operation.from_sequence(op1.at(1), op2.at(2))

        with pytest.raises(ValueError, match="Hilbert dimensionality of op.*"):
            Operation.from_sequence(op1.at(0), op_qutrit.at(0))

        circuit3q = Operation.from_sequence(op1.at(0), op2.at(1), op3.at(0, 1),
                          op1.at(1), op2.at(0), op3.at(0, 2))
        assert circuit3q.num_qubits == 3
        assert len(circuit3q.operations) == 6

        with pytest.raises(ValueError, match="Number of indices is not .*"):
            Operation.from_sequence(op1.at(0), op3.at(0))

        with pytest.raises(ValueError, match="Number of indices is not .*"):
            circuit3q.at(0, 1)

        circuit4q = Operation.from_sequence(op3.at(0, 2), circuit3q.at(1, 2, 3))
        assert len(circuit4q.operations) == 7
        assert circuit4q.operations[0].indices == (0, 2)
        for o1, o2 in zip(circuit4q.operations[1:], circuit3q.operations):
            assert np.all(np.array(o1.indices) == np.array(o2.indices) + 1)

        circuit4q = Operation.from_sequence(
            circuit3q.at(2, 0, 3), op3.at(0, 1), op2.at(1))
        assert len(circuit4q.operations) == 8
        assert circuit4q.operations[0].indices == (2,)
        assert circuit4q.operations[1].indices == (0,)
        assert circuit4q.operations[2].indices == (2, 0)

        Operation.from_sequence(
            circuit3q.at(0, 1, 2),
            Operation.from_sequence(op1, op2).at(1)
        )

    def test_chain_apply(self):
        b = (bases.general(2),) * 3
        dm = random_density_matrix(8, seed=93)
        pv1 = PauliVector.from_dm(dm, b)
        pv2 = PauliVector.from_dm(dm, b)

        # Some random gate sequence
        op_indices = [(lib2.rotate_x(np.pi/2), (0,)),
                      (lib2.rotate_y(0.3333), (1,)),
                      (lib2.cphase(), (0, 2)),
                      (lib2.cphase(), (1, 2)),
                      (lib2.rotate_x(-np.pi/2), (0,))]

        for op, indices in op_indices:
            op(pv1, *indices),

        circuit = Operation.from_sequence(
            *(op.at(*ix) for op, ix in op_indices))
        circuit(pv2, 0, 1, 2)

        assert np.all(pv1.to_pv() == pv2.to_pv())
