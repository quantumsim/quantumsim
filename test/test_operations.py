# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from copy import copy
from numpy import pi
from pytest import approx

import quantumsim.operations.qubits as lib2
import quantumsim.operations.qutrits as lib3
from quantumsim import bases, Operation
from quantumsim.algebra.tools import random_hermitian_matrix
from quantumsim.operations import ParametrizedOperation
from quantumsim.operations.operation import OperationNotDefinedError
from quantumsim.pauli_vectors import PauliVectorNumpy as PauliVector


class TestOperations:
    def test_kraus_to_ptm_qubit(self):
        p_damp = 0.5
        damp_kraus_mat = np.array(
            [[[1, 0], [0, np.sqrt(1 - p_damp)]],
             [[0, np.sqrt(p_damp)], [0, 0]]])

        gm_qubit_basis = (bases.gell_mann(2),)
        gm_two_qubit_basis = gm_qubit_basis * 2

        damp_op = Operation.from_kraus(damp_kraus_mat, gm_qubit_basis)
        damp_ptm = damp_op.ptm(gm_qubit_basis)

        expected_mat = np.array([[1, 0, 0, 0],
                                 [0, np.sqrt(1-p_damp), 0, 0],
                                 [0, 0, np.sqrt(1-p_damp), 0],
                                 [p_damp, 0, 0, 1-p_damp]])
        assert np.allclose(damp_ptm, expected_mat)

        with pytest.raises(ValueError, match=r'.* must be list-like, .*'):
            damp_op.set_bases(bases.gell_mann(2), bases.gell_mann(2))

        cz_kraus_mat = np.diag([1, 1, 1, -1])
        cz = Operation.from_kraus(cz_kraus_mat, gm_two_qubit_basis)
        cz_ptm = cz.ptm(gm_two_qubit_basis)

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

        cz = Operation.from_kraus(cz_kraus_mat, system_bases)
        cz_ptm = cz.ptm(system_bases)

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
        assert np.allclose(op1.ptm(gell_mann_basis), op2.ptm(gell_mann_basis))
        assert op1.bases_in == op2.bases_in
        assert op1.bases_out == op2.bases_out

    def test_chain_create(self):
        op1 = lib2.rotate_x()
        op2 = lib2.rotate_y()
        op3 = lib2.cnot()
        op_qutrit = lib3.rotate_x()

        circuit = Operation.from_sequence(op1.at(0), op2.at(0))
        assert circuit.num_qubits == 1
        assert len(circuit._units) == 2

        circuit = Operation.from_sequence(op1.at(1), op2.at(0))
        assert circuit.num_qubits == 2
        assert len(circuit._units) == 2

        with pytest.raises(ValueError, match=".* must form an ordered set .*"):
            Operation.from_sequence(op1.at(2), op2.at(0))

        with pytest.raises(ValueError, match=".* must form an ordered set .*"):
            Operation.from_sequence(op1.at(1), op2.at(2))

        with pytest.raises(ValueError, match="Hilbert dimensionality of op.*"):
            Operation.from_sequence(op1.at(0), op_qutrit.at(0))

        circuit3q = Operation.from_sequence(op1.at(0), op2.at(1), op3.at(0, 1),
                                            op1.at(1), op2.at(0), op3.at(0, 2))
        assert circuit3q.num_qubits == 3
        assert len(circuit3q._units) == 6

        with pytest.raises(ValueError, match="Number of indices is not .*"):
            Operation.from_sequence(op1.at(0), op3.at(0))

        with pytest.raises(ValueError, match="Number of indices is not .*"):
            circuit3q.at(0, 1)

        circuit4q = Operation.from_sequence(
            op3.at(0, 2), circuit3q.at(1, 2, 3))
        assert len(circuit4q._units) == 7
        assert circuit4q._units[0].indices == (0, 2)
        for o1, o2 in zip(circuit4q._units[1:], circuit3q._units):
            assert np.all(np.array(o1.indices) == np.array(o2.indices) + 1)

        circuit4q = Operation.from_sequence(
            circuit3q.at(2, 0, 3), op3.at(0, 1), op2.at(1))
        assert len(circuit4q._units) == 8
        assert circuit4q._units[0].indices == (2,)
        assert circuit4q._units[1].indices == (0,)
        assert circuit4q._units[2].indices == (2, 0)

        Operation.from_sequence(
            circuit3q.at(0, 1, 2),
            Operation.from_sequence(op1, op2).at(1)
        )

    def test_chain_apply(self):
        b = (bases.general(2),) * 3
        dm = random_hermitian_matrix(8, seed=93)
        pv1 = PauliVector.from_dm(dm, b)
        pv2 = PauliVector.from_dm(dm, b)

        # Some random gate sequence
        op_indices = [(lib2.rotate_x(np.pi/2), (0,)),
                      (lib2.rotate_y(0.3333), (1,)),
                      (lib2.cphase(), (0, 2)),
                      (lib2.cphase(), (1, 2)),
                      (lib2.rotate_x(-np.pi/2), (0,))]

        for op, indices in op_indices:
            op(pv1, *indices)

        circuit = Operation.from_sequence(
            *(op.at(*ix) for op, ix in op_indices))
        circuit(pv2, 0, 1, 2)
        assert np.all(pv1.to_pv() == pv2.to_pv())

    def test_ptm(self):
        # Some random gate sequence
        op_indices = [(lib2.rotate_x(np.pi/2), (0,)),
                      (lib2.rotate_y(0.3333), (1,)),
                      (lib2.cphase(), (0, 2)),
                      (lib2.cphase(), (1, 2)),
                      (lib2.rotate_x(-np.pi/2), (0,))]
        circuit = Operation.from_sequence(
            *(op.at(*ix) for op, ix in op_indices))

        b = (bases.general(2),) * 3
        ptm = circuit.ptm(b, b)
        assert isinstance(ptm, np.ndarray)

        op_3q = Operation.from_ptm(ptm, b)
        dm = random_hermitian_matrix(8, seed=93)
        state1 = PauliVector.from_dm(dm, b)
        state2 = PauliVector.from_dm(dm, b)

        circuit(state1, 0, 1, 2)
        op_3q(state2, 0, 1, 2)
        assert np.allclose(state1.to_pv(), state2.to_pv())

    def test_lindblad_singlequbit(self):
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
        op1 = Operation.from_lindblad_form(t1, b1, b2, hamiltonian=ham,
                                           lindblad_ops=lindblad_ops)
        op2 = Operation.from_lindblad_form(t2, b2, b1, hamiltonian=ham,
                                           lindblad_ops=lindblad_ops)
        op = Operation.from_lindblad_form(t1+t2, b1, hamiltonian=ham,
                                          lindblad_ops=lindblad_ops)
        dm = random_hermitian_matrix(2, seed=3)
        state1 = PauliVector.from_dm(dm, b1)
        state2 = PauliVector.from_dm(dm, b1)

        op1(state1, 0)
        op2(state1, 0)
        op(state2, 0)
        assert np.allclose(state1.to_pv(), state2.to_pv())

    def test_lindblad_time_inverse(self):
        ham = random_hermitian_matrix(2, seed=4)
        b = (bases.general(2),)
        op_plus = Operation.from_lindblad_form(20, b, hamiltonian=ham)
        op_minus = Operation.from_lindblad_form(20, b, hamiltonian=-ham)
        dm = random_hermitian_matrix(2, seed=5)
        state = PauliVector.from_dm(dm, b)
        op_plus(state, 0)
        op_minus(state, 0)
        assert np.allclose(state.to_dm(), dm)

    def test_lindblad_two_qubit(self):
        b = (bases.general(2),)
        iden = np.array([[1, 0], [0, 1]])
        ham1 = random_hermitian_matrix(2, seed=6)
        ham2 = random_hermitian_matrix(2, seed=7)
        ham = np.kron(ham1, iden).reshape(2, 2, 2, 2) + \
            np.kron(iden, ham2).reshape(2, 2, 2, 2)
        dm = random_hermitian_matrix(4, seed=3)
        op1 = Operation.from_lindblad_form(25, b, hamiltonian=ham1)
        op2 = Operation.from_lindblad_form(25, b, hamiltonian=ham2)
        op = Operation.from_lindblad_form(25, b*2, hamiltonian=ham)
        state1 = PauliVector.from_dm(dm, b*2)
        state2 = PauliVector.from_dm(dm, b*2)
        op1(state1, 0)
        op2(state1, 1)
        op(state2, 0, 1)
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
        ops = [np.kron(op, iden).reshape(2, 2, 2, 2) for op in ops1] + \
              [np.kron(iden, op).reshape(2, 2, 2, 2) for op in ops2]
        op1 = Operation.from_lindblad_form(25, b, lindblad_ops=ops1)
        op2 = Operation.from_lindblad_form(25, b, lindblad_ops=ops2)
        op = Operation.from_lindblad_form(25, b*2, lindblad_ops=ops)
        state1 = PauliVector.from_dm(dm, b*2)
        state2 = PauliVector.from_dm(dm, b*2)
        op1(state1, 0)
        op2(state1, 1)
        op(state2, 0, 1)
        assert np.allclose(state1.to_pv(), state2.to_pv())

        op1 = Operation.from_lindblad_form(25, b, hamiltonian=ham1,
                                           lindblad_ops=ops1)
        op2 = Operation.from_lindblad_form(25, b, hamiltonian=ham2,
                                           lindblad_ops=ops2)
        op = Operation.from_lindblad_form(25, b*2, hamiltonian=ham,
                                          lindblad_ops=ops)
        state1 = PauliVector.from_dm(dm, b*2)
        state2 = PauliVector.from_dm(dm, b*2)
        op1(state1, 0)
        op2(state1, 1)
        op(state2, 0, 1)
        assert np.allclose(state1.to_pv(), state2.to_pv())


class TestParametrizedOperations:
    def test_create(self):
        op_1q = lib2.rotate_x(0.5*pi)
        basis = (bases.general(2),)

        with pytest.raises(ValueError,
                           match=".*can't accept free arguments.*"):
            ParametrizedOperation(lambda *args: op_1q, basis, basis)
        with pytest.raises(ValueError,
                           match=".*can't accept free keyword arguments.*"):
            ParametrizedOperation(lambda **kwargs: op_1q, basis, basis)
        with pytest.raises(OperationNotDefinedError,
                           match="Operation placeholder does not have a PTM"):
            ParametrizedOperation(lambda: op_1q, basis).ptm(basis)
        with pytest.raises(OperationNotDefinedError,
                           match="Operation placeholder can not be called"):
            ParametrizedOperation(lambda: op_1q, basis)(PauliVector(basis))
