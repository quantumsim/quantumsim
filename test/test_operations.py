# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from copy import copy
from numpy import pi
from pytest import approx

from quantumsim import bases, Operation
from quantumsim.algebra.tools import random_density_matrix
from quantumsim.operations import ParametrizedOperation
from quantumsim.operations.operation import OperationNotDefinedError
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
        dm = random_density_matrix(8, seed=93)
        state1 = PauliVector.from_dm(dm, b)
        state2 = PauliVector.from_dm(dm, b)

        circuit(state1, 0, 1, 2)
        op_3q(state2, 0, 1, 2)
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

    def test_params_numbers(self):
        angle_ref = -0.8435
        basis = (bases.general(2),)

        op1q = ParametrizedOperation(lambda angle: lib2.rotate_y(angle), basis)
        assert op1q.params == {'angle'}

        op = op1q.substitute(angle=angle_ref)
        assert not isinstance(op, ParametrizedOperation)
        ptm_ref = lib2.rotate_y(angle=angle_ref).ptm(basis, basis)
        assert op.ptm(basis, basis) == approx(ptm_ref)

        op = op1q.substitute(foo=42, bar='baz')
        assert op.params == {'angle'}
        op = op.substitute(angle=angle_ref, extra_param=42)
        assert not isinstance(op, ParametrizedOperation)
        assert op.ptm(basis, basis) == approx(ptm_ref)

        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib2.rotate_y(angle_rotate).at(1),
                lib2.cphase(angle_cphase).at(0, 1),
                lib2.rotate_y(-angle_rotate).at(1))

        op2q = ParametrizedOperation(cnot_like, basis*2)

        params = dict(angle_cphase=1.02 * pi, angle_rotate=0.47 * pi, foo='bar')
        ptm_ref = cnot_like(params['angle_cphase'], params['angle_rotate']) \
            .ptm(basis * 2, basis * 2)
        assert op2q.params == {'angle_cphase', 'angle_rotate'}
        op = op2q.substitute(**params)
        assert not isinstance(op, ParametrizedOperation)
        assert op.ptm(basis * 2, basis * 2) == approx(ptm_ref)

        _ = ParametrizedOperation(lambda: lib2.rotate_z(0.5*pi), basis) \
            .substitute()
        with pytest.raises(RuntimeError,
                           match=".*does not match one of the basis.*"):
            ParametrizedOperation(lambda: lib2.rotate_z(0.5*pi),
                                  basis*2, basis*2).substitute()

    def test_params_rename(self):
        def cnot_like(angle_cphase, angle_rotate):
            return Operation.from_sequence(
                lib2.rotate_y(angle_rotate).at(1),
                lib2.cphase(angle_cphase).at(0, 1),
                lib2.rotate_y(-angle_rotate).at(1))

        angle_cphase_ref = 0.98 * pi
        angle_rotate_ref = 0.5 * pi
        basis = (bases.general(2),) * 2
        op = ParametrizedOperation(cnot_like, basis, basis)
        ptm_ref = cnot_like(angle_cphase_ref, angle_rotate_ref) \
            .ptm(basis, basis)

        op = op.substitute(angle_cphase='foo')
        assert op.params == {'foo', 'angle_rotate'}
        assert op.substitute(
            foo=angle_cphase_ref, angle_rotate=angle_rotate_ref
        ).ptm(basis, basis) == approx(ptm_ref)

        op = op.substitute(foo='bar')
        assert op.params == {'bar', 'angle_rotate'}
        assert op.substitute(
            bar=angle_cphase_ref, angle_rotate=angle_rotate_ref
        ).ptm(basis, basis) == approx(ptm_ref)

        op = op.substitute(bar=angle_cphase_ref)
        assert op.params == {'angle_rotate'}

        assert op.substitute(
            angle_rotate=angle_rotate_ref, angle_cphase=42., foo=12,
            bar='and now something completely different'
        ).ptm(basis, basis) == approx(ptm_ref)

        op = ParametrizedOperation(cnot_like, basis, basis)
        op = op.substitute(angle_cphase='foo', angle_rotate='bar')
        assert op.substitute(
            bar=angle_rotate_ref, foo=angle_cphase_ref,
            angle_cphase=-1, angle_rotate=-2
        ).ptm(basis, basis) == approx(ptm_ref)

        with pytest.raises(ValueError,
                           match=".* not a valid Python identifier."):
            op.substitute(bar='')
        with pytest.raises(ValueError,
                           match=".* not a valid Python identifier."):
            op.substitute(bar='42')
