# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np
from numpy import pi
from pytest import approx

from quantumsim import bases, State
from quantumsim.algebra.tools import random_hermitian_matrix
from quantumsim.circuits import optimize, Gate, Circuit

from quantumsim.models.perfect import qubits as lib2
from quantumsim.models.perfect import qutrits as lib3


# FIXME All checks that check len(gate.free_parameters) should instead check if
#       the gate has a PTM
class TestCompiler:
    def test_opt_basis_single_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b1 = b.subbasis([1])
        b01 = b.computational_subbasis()

        # Identity up to floating point error
        rot = optimize(lib2.rotate_x('Q')(angle=2 * np.pi), bases_in=(b0,))
        assert rot.bases_in == (b0,)
        assert rot.bases_out == (b0,)
        rot = optimize(lib2.rotate_x('Q')(angle=2 * np.pi), bases_in=(b1,))
        assert rot.bases_in == (b1,)
        assert rot.bases_out == (b1,)

        # RX(pi)
        rot = optimize(lib2.rotate_x('Q')(angle=np.pi), bases_in=(b0,))
        assert rot.bases_in == (b0,)
        assert rot.bases_out == (b1,)
        rot = optimize(lib2.rotate_x('Q')(angle=np.pi), bases_in=(b1,))
        assert rot.bases_in == (b1,)
        assert rot.bases_out == (b0,)

        # RY(pi/2)
        rot = optimize(lib2.rotate_y('Q')(angle=np.pi / 2), bases_in=(b01,))
        assert rot.bases_in[0] == b01
        assert rot.bases_out[0].dim_pauli == 3
        assert '0' in rot.bases_out[0].labels
        assert '1' in rot.bases_out[0].labels
        assert 'X10' in rot.bases_out[0].labels

    def test_opt_basis_two_qubit_2d(self):
        op = lib2.cnot('A', 'B')

        # Classical input basis -> classical output basis
        # Possible flip in control bit
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.subbasis([0, 1])
        b_in = (b01, b0)
        op_c = optimize(op, bases_in=b_in)
        assert op_c.bases_in[0] == b01
        assert op_c.bases_in[1] == b0
        assert op_c.bases_out[0] == b01
        assert op_c.bases_out[1] == b01

        # Classical control bit is not violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b0, b)
        op_c = optimize(op, bases_in=b_in)
        assert op_c.bases_in[0] == b0
        assert op_c.bases_in[1] == b
        assert op_c.bases_out[0] == b0
        assert op_c.bases_out[1] == b

        # Classical target bit will become quantum for quantum control bit,
        # input should not be violated
        b = bases.general(2)
        b0 = b.subbasis([0])
        b_in = (b, b0)
        op_c = optimize(op, bases_in=b_in)
        assert op_c.bases_in[0] == b
        assert op_c.bases_in[1] == b0
        assert op_c.bases_out[0] == b
        assert op_c.bases_out[1] == b

    def test_compile_single_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.computational_subbasis()

        op = lib2.rotate_y('Q')(angle=np.pi)
        assert op.shape == (4, 4)
        op_full = optimize(op, bases_in=(b,))
        assert op_full.shape == (4, 4)
        op_cl = optimize(op, bases_in=(b01,))
        assert op_cl.shape == (2, 2)

        op = lib2.rotate_x('Q')(angle=np.pi / 3)
        assert op.shape == (4, 4)
        op_full = optimize(op, bases_in=(b,), bases_out=(b01,))
        # X component of a state is irrelevant for the output.
        assert op_full.shape == (2, 3)
        op_cl = optimize(op, bases_in=(b0,))
        assert op_cl.shape == (3, 1)

    def test_compile_two_qubit_2d(self):
        b = bases.general(2)
        b0 = b.subbasis([0])
        b01 = b.computational_subbasis()

        op = lib2.cnot(0, 1)
        assert op.shape == (4, 4, 4, 4)
        op_full = optimize(op, bases_in=(b, b))
        assert op_full.shape == (4, 4, 4, 4)
        op_cl = optimize(op, bases_in=(b01, b01))
        assert op_cl.shape == (2, 2, 2, 2)
        op_cl = optimize(op, bases_in=(b0, b))
        assert op_cl.shape == (1, 4, 1, 4)

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_circ_compile_single_qubit(self, d, lib):
        b = bases.general(d)
        dm = random_hermitian_matrix(d, seed=487)

        bases_full = (b,)
        subbases = (b.subbasis([0, 1]),)
        angle = np.pi/5
        circ0 = lib.rotate_x(0)(angle=angle) + lib.rotate_x(0)(angle=angle)
        pv0 = State.from_dm([0], dm, bases_full)
        circ0 @ pv0
        assert len(circ0.qubits) == 1
        assert len(circ0.gates) == 2

        circ0_c = optimize(circ0, bases_full, bases_full)
        assert isinstance(circ0_c, Gate)
        assert len(circ0_c.free_parameters) == 0
        assert len(circ0_c.qubits) == 1
        pv1 = State.from_dm([0], dm, bases_full)
        circ0_c @ pv1
        op_angle = circ0_c
        op_2angle = optimize(lib.rotate_x(0)(angle=2*angle), bases_full, bases_full)
        assert isinstance(op_2angle, Gate)
        # FIXME assert op_angle.shape == op_2angle.shape
        assert op_angle.bases_in == op_2angle.bases_in
        assert op_angle.bases_out == op_2angle.bases_out
        assert op_angle.ptm == approx(op_2angle.ptm)
        assert pv1.to_pv() == approx(pv0.to_pv())

        circ_2pi = lib.rotate_x(0)(angle=np.pi) + lib.rotate_x(0)(angle=np.pi)
        circ_2pi_c1 = optimize(circ_2pi, subbases, bases_full)
        assert isinstance(circ_2pi_c1, Gate)
        assert len(circ0_c.free_parameters) == 0
        assert circ_2pi_c1.bases_in == subbases
        assert circ_2pi_c1.bases_out == subbases

        circ_2pi_c2 = optimize(circ_2pi, bases_full, subbases)
        assert isinstance(circ_2pi_c2, Gate)
        assert len(circ0_c.free_parameters) == 0
        assert circ_2pi_c2.bases_in == subbases
        assert circ_2pi_c2.bases_out == subbases

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_circ_merge_next(self, d, lib):
        b = bases.general(d)

        dm = random_hermitian_matrix(d ** 2, seed=574)

        circ = lib.rotate_x(0)(angle=np.pi / 5) + (
            lib.cphase(0, 1)(angle=3*np.pi/7, leakage_rate=0.1)
            if d == 3 else lib.cphase(0, 1)(angle=3*np.pi / 7)
        )

        bases_full = (b, b)
        circ_c = optimize(circ, bases_full, bases_full)
        assert len(circ.gates) == 2
        assert isinstance(circ_c, Gate)

        pv1 = State.from_dm([0, 1], dm, bases_full)
        pv2 = State.from_dm([0, 1], dm, bases_full)
        circ @ pv1
        circ_c @ pv2

        assert pv1.meas_prob(0) == approx(pv2.meas_prob(0))
        assert pv1.meas_prob(1) == approx(pv2.meas_prob(1))

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_circ_merge_prev(self, d, lib):
        b = bases.general(d)
        dm = random_hermitian_matrix(d*d, 4242)

        circ = (lib.cphase(0, 1)(angle=np.pi/7, leakage_rate=0.25)
                 if d == 3 else lib.cphase(0, 1)(angle=3*np.pi / 7)
                 ) + lib.rotate_x(0)(angle=4 * np.pi / 7)

        bases_full = (b, b)
        circ_c = optimize(circ, bases_full, bases_full)
        assert len(circ.gates) == 2
        assert isinstance(circ_c, Gate)

        pv1 = State.from_dm([0, 1], dm, bases_full)
        pv2 = State.from_dm([0, 1], dm, bases_full)
        circ @ pv1
        circ_c @ pv2

        assert np.allclose(pv1.meas_prob(0), pv2.meas_prob(0))
        assert np.allclose(pv1.meas_prob(1), pv2.meas_prob(1))

    @pytest.mark.parametrize('d,lib', [(2, lib2), (3, lib3)])
    def test_circ_compile_three_qubit(self, d, lib):
        b = bases.general(d)
        b0 = b.subbasis([0])

        circ0 = (
            lib.rotate_x('2')(angle=0.5*np.pi) +
            lib.cphase('0', '2')(angle=np.pi) +
            lib.cphase('1', '2')(angle=np.pi) +
            lib.rotate_x('2')(angle=-0.75*np.pi) +
            lib.rotate_x('2')(angle=0.25*np.pi)
        )
        circ1 = optimize(circ0, (b, b, b0), (b, b, b), ['0', '1', '2'])
        assert isinstance(circ1, Circuit)
        assert circ1.gates[0].qubits == ('0', '2')
        assert circ1.gates[0].bases_in == (b, b0)
        assert circ1.gates[0].bases_out[0] == b
        assert circ1.gates[1].qubits == ('1', '2')
        assert circ1.gates[1].bases_in[0] == b
        assert circ1.gates[1].bases_out[0] == b
        for label in '0', '1', 'X10', 'Y10':
            assert label in circ1.gates[1].bases_out[1].labels

    def test_circ_compile_leaking(self):
        b = bases.general(3)
        circ0 = (
            lib3.rotate_x('C')(angle=0.5*np.pi) +
            lib3.cphase('A', 'C')(angle=np.pi, leakage_rate=0.1) +
            lib3.cphase('B', 'C')(angle=np.pi, leakage_rate=0.1) +
            lib3.rotate_x('C')(angle=-0.75*np.pi) +
            lib3.rotate_x('C')(angle=0.25*np.pi)
        )
        b0 = b.subbasis([0])
        b01 = b.subbasis([0, 1])
        b0134 = b.subbasis([0, 1, 3, 4])
        circ1 = optimize(circ0, (b0, b0, b0134), (b, b, b), qubits=['A', 'B', 'C'])
        assert isinstance(circ1, Circuit)
        # Ancilla is not leaking here
        anc_basis = circ1.gates[1].bases_out[1]
        for label in anc_basis.labels:
            assert '2' not in label

        circ2 = optimize(circ0, (b01, b01, b0134), (b, b, b), qubits=['A', 'B', 'C'])
        # Ancilla is leaking here
        assert isinstance(circ2, Circuit)
        anc_basis = circ2.gates[1].bases_out[1]
        for label in '2', 'X20', 'Y20', 'X21', 'Y21':
            assert label in anc_basis.labels

    def test_zz_parity_compilation(self):
        b_full = bases.general(3)
        b0 = b_full.subbasis([0])
        b01 = b_full.subbasis([0, 1])
        b012 = b_full.subbasis([0, 1, 2])

        bases_in = (b01, b01, b0)
        bases_out = (b_full, b_full, b012)
        zz = (
            lib3.rotate_x(2)(angle=-pi/2) +
            lib3.cphase(0, 2)(angle=pi, leakage_rate=0.1) +
            lib3.cphase(2, 1)(angle=pi, leakage_rate=0.25) +
            lib3.rotate_x(2)(angle=pi/2) +
            lib3.rotate_x(0)(angle=pi) +
            lib3.rotate_x(1)(angle=pi)
        )
        # FIXME: implement Circuit.ptm()
        # zz_ptm = zz.ptm(bases_in, bases_out)
        zzc = optimize(zz, bases_in=bases_in, bases_out=bases_out, qubits=(0, 1, 2))
        # zzc_ptm = zzc.ptm(bases_in, bases_out)
        # assert zz_ptm == approx(zzc_ptm)

        units = list(zzc.operations())
        assert len(units) == 2
        op1 = units[0]
        op2 = units[1]
        assert op1.qubits == (0, 2)
        assert op2.qubits == (1, 2)
        assert op1.bases_in[0] == bases_in[0]
        assert op2.bases_in[0] == bases_in[1]
        assert op1.bases_in[1] == bases_in[2]
        # Qubit 0 did not leak
        assert op1.bases_out[0] == bases_out[0].subbasis([0, 1, 3, 4])
        # Qubit 1 leaked
        assert op2.bases_out[0] == bases_out[1].subbasis([0, 1, 2, 6])
        # Qubit 2 is measured
        assert op2.bases_out[1] == bases_out[2]

        dm = random_hermitian_matrix(3 ** 3, seed=85)
        state1 = State.from_dm([0, 1, 2], dm, (b01, b01, b0))
        state2 = State.from_dm([0, 1, 2], dm, (b01, b01, b0))

        zz @ state1
        zzc @ state2

        # Compiled version still needs to be projected, so we can't compare
        # Pauli vectors, so we can to check only DM diagonals.
        assert np.allclose(state1.diagonal, state2.diagonal)

    def test_compilation_with_placeholders(self):
        b_full = bases.general(3)
        b0 = b_full.subbasis([0])
        b01 = b_full.subbasis([0, 1])
        b012 = b_full.subbasis([0, 1, 2])

        bases_in = (b01, b01, b0)
        bases_out = (b_full, b_full, b012)
        angle1 = -np.pi/2
        angle2 = 5*pi/6
        phase02 = 1.15*pi
        phase21 = -0.93*pi
        zz = optimize((
            lib3.rotate_x(2)(angle=angle1) +
            lib3.cphase(0, 2)(angle=phase02) +
            lib3.cphase(2, 1)(angle=phase21) +
            lib3.rotate_x(2)(angle=angle2) +
            lib3.rotate_x(0)(angle=np.pi) +
            lib3.rotate_x(1)(angle=np.pi)
        ), bases_in, bases_out, qubits=[0, 1, 2])
        # FIXME implement PTM extraction
        # ptm_ref = zz.ptm(bases_in, bases_out)

        zz_parametrized = (
            lib3.rotate_x(2)(angle='angle1') +
            lib3.cphase(0, 2)(angle='phase02') +
            lib3.cphase(2, 1)(angle='phase21') +
            lib3.rotate_x(2)(angle='angle2') +
            lib3.rotate_x(0)(angle=np.pi) +
            lib3.rotate_x(1)(angle=np.pi)
        )
        zzpc = optimize(zz_parametrized, bases_in, bases_out, qubits=[0, 1, 2])
        assert isinstance(zzpc, Circuit)
        assert len(list(zzpc.operations())) == 6

        zz_parametrized = (
                lib3.rotate_x(2)(angle='angle1') +
                lib3.cphase(0, 2)(angle='phase02') +
                lib3.cphase(2, 1)(angle=phase21) +
                lib3.rotate_x(2)(angle=angle2) +
                lib3.rotate_x(0)(angle=pi) +
                lib3.rotate_x(1)(angle=pi)
        )
        zzpc = optimize(zz_parametrized, bases_in, bases_out, qubits=[0, 1, 2])
        assert len(list(zzpc.operations())) == 4
        zz_new = zz_parametrized(angle1=angle1, phase02=phase02, foo='bar')
        zzpc = optimize(zz_new, bases_in, bases_out, qubits=[0, 1, 2])
        assert len(list(zzpc.operations())) == 2
        # FIXME implement PTM extraction
        # assert zzpc.ptm(bases_in, bases_out) == approx(ptm_ref)
