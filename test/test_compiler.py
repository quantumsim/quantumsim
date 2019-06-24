# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
import warnings
from pytest import approx

from quantumsim import bases, Operation
from quantumsim.algebra.tools import random_density_matrix
# noinspection PyProtectedMember
from quantumsim.operations.operation import _PTMOperation
from quantumsim.states import StateNumpy as State


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from quantumsim.models import qubits as lib2
    from quantumsim.models import transmons as lib3


class TestCompiler:
    def test_opt_basis_single_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b1 = b.subbasis([1])
        b01 = b.computational_subbasis()

        # Identity up to floating point error
        rot = lib2.rotate_x(2 * np.pi).compile(bases_in=(b0,))
        assert rot.bases_in == (b0,)
        assert rot.bases_out == (b0,)
        rot = lib2.rotate_x(2 * np.pi).compile(bases_in=(b1,))
        assert rot.bases_in == (b1,)
        assert rot.bases_out == (b1,)

        # RX(pi)
        rot = lib2.rotate_x(np.pi).compile(bases_in=(b0,))
        assert rot.bases_in == (b0,)
        assert rot.bases_out == (b1,)
        rot = lib2.rotate_x(np.pi).compile(bases_in=(b1,))
        assert rot.bases_in == (b1,)
        assert rot.bases_out == (b0,)

        # RY(pi/2)
        rot = lib2.rotate_y(np.pi / 2).compile(bases_in=(b01,))
        assert rot.bases_in[0] == b01
        assert rot.bases_out[0].dim_pauli == 3
        assert '0' in rot.bases_out[0].labels
        assert '1' in rot.bases_out[0].labels
        assert 'X10' in rot.bases_out[0].labels

    def test_opt_basis_two_qubit_2d(self):
        op = lib2.cnot()

        # Classical input basis -> classical output basis
        # Possible flip in control bit
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.subbasis([0, 1])
        b_in = (b01, b0)
        op_c = op.compile(bases_in=b_in)
        assert op_c.bases_in[0] == b01
        assert op_c.bases_in[1] == b0
        assert op_c.bases_out[0] == b01
        assert op_c.bases_out[1] == b01

        # Classical control bit is not violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b0, b)
        op_c = op.compile(bases_in=b_in)
        assert op_c.bases_in[0] == b0
        assert op_c.bases_in[1] == b
        assert op_c.bases_out[0] == b0
        assert op_c.bases_out[1] == b

        # Classical target bit will become quantum for quantum control bit,
        # input should not be violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b, b0)
        op_c = op.compile(bases_in=b_in)
        assert op_c.bases_in[0] == b
        assert op_c.bases_in[1] == b0
        assert op_c.bases_out[0] == b
        assert op_c.bases_out[1] == b

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

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_compile_single_qubit(self, d, lib):
        b = bases.general(d)
        dm = random_density_matrix(d, seed=487)

        bases_full = (b,)
        subbases = (b.subbasis([0, 1]),)
        angle = np.pi/5
        rx_angle = lib.rotate_x(angle)
        rx_2angle = lib.rotate_x(2*angle)
        chain0 = Operation.from_sequence(rx_angle.at(0), rx_angle.at(0))
        state0 = State.from_dm(dm, bases_full)
        chain0(state0, 0)
        assert chain0.num_qubits == 1
        assert len(chain0.operations) == 2

        chain0_c = chain0.compile(bases_full, bases_full)
        state1 = State.from_dm(dm, bases_full)
        chain0_c(state1, 0)
        assert chain0_c.num_qubits == 1
        assert isinstance(chain0_c, _PTMOperation)
        op_angle = chain0_c
        op_2angle = rx_2angle.compile(bases_full, bases_full)
        assert op_angle.shape == op_2angle.shape
        assert op_angle.bases_in == op_2angle.bases_in
        assert op_angle.bases_out == op_2angle.bases_out
        assert op_angle.ptm == approx(op_2angle.ptm)
        assert state1.to_pv() == approx(state0.to_pv())

        rx_pi = lib.rotate_x(np.pi)
        chain_2pi = Operation.from_sequence(rx_pi.at(0), rx_pi.at(0))
        chain_2pi_c1 = chain_2pi.compile(subbases, bases_full)
        assert isinstance(chain_2pi_c1, _PTMOperation)
        assert chain_2pi_c1.bases_in == subbases
        assert chain_2pi_c1.bases_out == subbases

        chain_2pi_c2 = chain_2pi.compile(bases_full, subbases)
        assert isinstance(chain_2pi_c2, _PTMOperation)
        assert chain_2pi_c2.bases_in == subbases
        assert chain_2pi_c2.bases_out == subbases

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_merge_next(self, d, lib):
        b = bases.general(d)

        dm = random_density_matrix(d**2, seed=574)

        chain = Operation.from_sequence(
            lib.rotate_x(np.pi / 5).at(0),
            (lib.cphase(angle=3*np.pi/7, leakage_rate=0.1)
             if d == 3 else lib.cphase(3*np.pi / 7)).at(0, 1),
        )

        bases_full = (b, b)
        chain_c = chain.compile(bases_full, bases_full)
        assert len(chain.operations) == 2
        assert isinstance(chain_c, _PTMOperation)

        state1 = State.from_dm(dm, bases_full)
        state2 = State.from_dm(dm, bases_full)
        chain(state1, 0, 1)
        chain_c(state2, 0, 1)

        assert state1.meas_prob(0) == approx(state2.meas_prob(0))
        assert state1.meas_prob(1) == approx(state2.meas_prob(1))

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_merge_prev(self, d, lib):
        b = bases.general(d)

        rng = np.random.RandomState(4242)
        dm = rng.randn(d*d, d*d) + 1j * rng.randn(d*d, d*d)
        dm += dm.conjugate().transpose()
        dm /= dm.trace()

        chain = Operation.from_sequence(
            (lib.cphase(angle=np.pi/7, leakage_rate=0.25)
             if d == 3 else lib.cphase(3*np.pi / 7)).at(0, 1),
            lib.rotate_x(4 * np.pi / 7).at(0),
        )

        bases_full = (b, b)
        chain_c = chain.compile(bases_full, bases_full)
        assert len(chain.operations) == 2
        assert isinstance(chain_c, _PTMOperation)

        state1 = State.from_dm(dm, bases_full)
        state2 = State.from_dm(dm, bases_full)
        chain(state1, 0, 1)
        chain_c(state2, 0, 1)

        assert np.allclose(state1.meas_prob(0), state2.meas_prob(0))
        assert np.allclose(state1.meas_prob(1), state2.meas_prob(1))

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_chain_compile_three_qubit(self, d, lib):
        b = bases.general(d)
        b0 = b.subbasis([0])

        chain0 = Operation.from_sequence(
            lib.rotate_x(0.5*np.pi).at(2),
            lib.cphase().at(0, 2),
            lib.cphase().at(1, 2),
            lib.rotate_x(-0.75*np.pi).at(2),
            lib.rotate_x(0.25*np.pi).at(2),
        )
        chain1 = chain0.compile((b, b, b0), (b, b, b))
        assert chain1.operations[0].indices == (0, 2)
        assert chain1.operations[0].operation.bases_in == (b, b0)
        assert chain1.operations[0].operation.bases_out[0] == b
        assert chain1.operations[1].indices == (1, 2)
        assert chain1.operations[1].operation.bases_in[0] == b
        assert chain1.operations[1].operation.bases_out[0] == b
        for label in '0', '1', 'X10', 'Y10':
            assert label in chain1.operations[1].operation.bases_out[1].labels

    def test_chain_compile_leaking(self):
        b = bases.general(3)
        chain0 = Operation.from_sequence(
            lib3.rotate_x(0.5*np.pi).at(2),
            lib3.cphase(leakage_rate=0.1).at(0, 2),
            lib3.cphase(leakage_rate=0.1).at(1, 2),
            lib3.rotate_x(-0.75*np.pi).at(2),
            lib3.rotate_x(0.25*np.pi).at(2),
        )
        b0 = b.subbasis([0])
        b01 = b.subbasis([0, 1])
        b0134 = b.subbasis([0, 1, 3, 4])
        chain1 = chain0.compile((b0, b0, b0134), (b, b, b))
        # Ancilla is not leaking here
        anc_basis = chain1.operations[1].operation.bases_out[1]
        for label in anc_basis.labels:
            assert '2' not in label

        chain2 = chain0.compile((b01, b01, b0134), (b, b, b))
        # Ancilla is leaking here
        anc_basis = chain2.operations[1].operation.bases_out[1]
        for label in '2', 'X20', 'Y20', 'X21', 'Y21':
            assert label in anc_basis.labels

    def test_zz_parity_compilation(self):
        b_full = bases.general(3)
        b0 = b_full.subbasis([0])
        b01 = b_full.subbasis([0, 1])
        b012 = b_full.subbasis([0, 1, 2])

        bases_in = (b01, b01, b0)
        bases_out = (b_full, b_full, b012)
        zz = Operation.from_sequence(
            lib3.rotate_x(-np.pi/2).at(2),
            lib3.cphase(leakage_rate=0.1).at(0, 2),
            lib3.cphase(leakage_rate=0.25).at(2, 1),
            lib3.rotate_x(np.pi/2).at(2),
            lib3.rotate_x(np.pi).at(0),
            lib3.rotate_x(np.pi).at(1)
        )
        zzc = zz.compile(bases_in=bases_in, bases_out=bases_out)

        assert len(zzc.operations) == 2
        op1, ix1 = zzc.operations[0]
        op2, ix2 = zzc.operations[1]
        assert ix1 == (0, 2)
        assert ix2 == (1, 2)
        assert op1.bases_in[0] == bases_in[0]
        assert op2.bases_in[0] == bases_in[1]
        assert op1.bases_in[1] == bases_in[2]
        # Qubit 0 did not leak
        assert op1.bases_out[0] == bases_out[0].subbasis([0, 1, 3, 4])
        # Qubit 1 leaked
        assert op2.bases_out[0] == bases_out[1].subbasis([0, 1, 2, 6])
        # Qubit 2 is measured
        assert op2.bases_out[1] == bases_out[2]

        dm = random_density_matrix(3**3, seed=85)
        state1 = State.from_dm(dm, (b01, b01, b0))
        state2 = State.from_dm(dm, (b01, b01, b0))

        zz(state1, 0, 1, 2)
        zzc(state2, 0, 1, 2)

        # Compiled version still needs to be projected, so we can't compare
        # Pauli vectors, so we can to check only DM diagonals.
        assert np.allclose(state1.diagonal(), state2.diagonal())
