import numpy as np
import pytest
import sympy

from numpy import pi
from pytest import approx
from quantumsim.algebra import kraus_to_ptm
from quantumsim.algebra.tools import random_unitary_matrix
from quantumsim.circuits import Gate, allow_param_repeat
from quantumsim import bases


def ptm_cphase(angle):
    return kraus_to_ptm(np.array([[[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, np.exp(1j * angle)]]]), bases2q, bases2q)


def ptm_rotate(angle):
    sin, cos = np.sin(angle / 2), np.cos(angle / 2)
    return kraus_to_ptm(np.array([[[cos, -1j*sin], [-1j*sin, cos]]]),
                        (bases.general(2),), (bases.general(2),))


bases1q = (bases.general(2),)
bases2q = bases1q * 2
ptm_cnot = kraus_to_ptm(np.array([[[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]]), bases2q, bases2q)
ptm_cz = ptm_cphase(pi)


class TestCircuitsCommon:
    def test_gate_create_no_params(self):
        dim = 2

        ptm_1q = kraus_to_ptm(random_unitary_matrix(dim, 5001).reshape(1, dim, dim),
                              bases1q, bases1q)
        ptm_2q = kraus_to_ptm(
            random_unitary_matrix(dim*2, 5002).reshape(1, dim*2, dim*2),
            bases2q, bases2q)

        gate = Gate('qubit', dim, lambda: (ptm_1q, bases1q, bases1q))
        assert gate.qubits == ('qubit',)
        assert len(gate.free_parameters) == 0
        assert gate.set_bases(bases1q, bases1q).ptm == approx(ptm_1q)

        subbasis = bases.general(2).subbasis([0, 1]),
        gate = Gate(('Q1',), dim, lambda: (ptm_1q, bases1q, bases1q),
                    bases_in=subbasis, bases_out=subbasis)
        assert len(gate.free_parameters) == 0
        assert gate.ptm == approx(ptm_1q[:2, :2])
        # If subbasis was provided, it must trunkate the PTM, that shouls also
        # persist on upcasting basis back.
        assert gate.set_bases(bases1q, bases1q).ptm != approx(ptm_1q)

        gate = Gate(('D', 'A'), dim, lambda: (ptm_2q, bases2q,  bases2q))
        assert gate.qubits == ('D', 'A')
        assert len(gate.free_parameters) == 0
        assert gate.set_bases(bases2q, bases2q).ptm == approx(ptm_2q)

    def test_gate_params_call(self):
        dim = 2
        angle1, angle2 = sympy.symbols('angle1 angle2')

        def rotate_cnot_rotate(angle1, angle2):
            cnot = ptm_cnot
            return (np.einsum('ai,ijkl,km->ajml',
                              ptm_rotate(angle2), cnot, ptm_rotate(angle1),
                              optimize=True),
                    bases2q, bases2q)

        angle1_1 = 1.2 * pi
        angle2_1 = 0.6 * pi
        angle1_2 = 0.8 * pi
        angle2_2 = 0.3 * pi

        gate = Gate(('D', 'A'), dim, rotate_cnot_rotate)
        assert gate.free_parameters == {angle1, angle2}

        gate1 = gate(angle1=angle1_1, angle2=angle2_1)
        assert gate.free_parameters == {angle1, angle2}
        assert gate1.free_parameters == set()

        gate2 = gate(angle1=angle1_2)
        assert gate.free_parameters == {angle1, angle2}
        assert gate2.free_parameters == {angle2}

        assert gate1.ptm == approx(
            gate(angle1=angle1_1, angle2=angle2_1).ptm)
        # angle1 has already been set for gate2
        assert gate2(angle2=angle2_2, angle1=0xdeadbeef).ptm == approx(
            gate(angle1=angle1_2, angle2=angle2_2).ptm)

    @pytest.mark.xfail
    def test_circuits_add(self):
        dim = 2
        ptm_rplus = ptm_rotate(0.5 * pi)
        ptm_rminus = ptm_rotate(-0.5 * pi)
        grplus = Gate('Q0', dim, lambda: (ptm_rplus, bases1q, bases1q))
        grminus = Gate('Q0', dim, lambda: (ptm_rminus, bases1q, bases1q))
        gcphase = Gate(('Q0', 'Q1'), dim, lambda: (ptm_cz, bases2q, bases2q))
        basis = (bases.general(2),) * 2

        circuit = grplus + gcphase
        ptm = np.einsum('ijkl, km -> ijml', ptm_cz, ptm_rplus)
        assert circuit.qubits == ['Q0', 'Q1']
        assert len(circuit.gates) == 2
        assert circuit.ptm == approx(ptm)

        circuit = circuit + grminus
        ptm = np.einsum('ai, ijkl -> ajkl', ptm_rminus, ptm)
        assert circuit.qubits == ['Q0', 'Q1']
        assert len(circuit.gates) == 3
        assert circuit.ptm == approx(ptm)

        circuit = grplus + (gcphase + grminus)
        ptm = np.einsum('ai, ijkl, km -> ajml', ptm_rminus, ptm_cz, ptm_rplus)
        assert circuit.qubits == ['Q0', 'Q1']
        assert len(circuit.gates) == 3
        assert circuit.ptm == approx(ptm)

        grplus = Gate('Q1', dim, lambda: (ptm_rplus, bases1q, bases1q))
        grminus = Gate('Q1', dim, lambda: (ptm_rminus, bases1q, bases1q))
        circuit = grplus + gcphase + grminus
        ptm = np.einsum('ai, ijkl, km -> ajml', ptm_rminus, ptm_cz, ptm_rplus)
        assert circuit.qubits == ['Q1', 'Q0']
        assert len(circuit.gates) == 3
        assert circuit.ptm == approx(ptm)

        basis = (basis[0],) * 3
        grplus = Gate('Q2', dim, lambda: (ptm_rplus, bases1q, bases1q))
        grminus = Gate('Q0', dim, lambda: (ptm_rminus, bases1q, bases1q))
        circuit = grplus + gcphase + grminus
        ptm = np.einsum('mi, ijkl, ab -> amjbkl', ptm_rminus, ptm_cz, ptm_rplus)
        assert circuit.qubits == ['Q2', 'Q0', 'Q1']
        assert len(circuit.gates) == 3
        assert circuit.ptm == approx(ptm)

        grplus = Gate('Q0', dim, lambda: (ptm_rplus, bases1q, bases1q))
        grminus = Gate('Q2', dim, lambda: (ptm_rminus, bases1q, bases1q))
        circuit = grplus + gcphase + grminus
        ptm = np.einsum('ab, ijkl, kn -> ijanlb', ptm_rminus, ptm_cz, ptm_rplus)
        assert circuit.qubits == ['Q0', 'Q1', 'Q2']
        assert len(circuit.gates) == 3
        assert circuit.ptm(basis, basis) == approx(ptm)

    @pytest.mark.xfail
    def test_circuits_params(self):
        dim = 2
        grotate = Gate('Q0', dim, lambda angle: (ptm_rotate(angle), bases1q, bases1q))
        gcphase = Gate(('Q0', 'Q1'), dim, lambda angle: (ptm_cphase(angle), bases2q,
                                                         bases2q))

        with pytest.raises(RuntimeError,
                           match=r".*free parameters.*\n"
                                 r".*angle.*\n"
                                 r".*allow_param_repeat.*"):
            _ = gcphase + grotate

        with allow_param_repeat():
            circuit = grotate + gcphase + grotate

        assert circuit.free_parameters == {sympy.symbols('angle')}
        assert len(circuit.gates) == 3
        angle = 0.736
        assert circuit(angle=angle).ptm(bases2q, bases2q) == approx(
            np.einsum('ai, ijkl, km -> ajml',
                      ptm_rotate(angle), ptm_cphase(angle), ptm_rotate(angle)))

        angle1 = 0.4 * pi
        angle2 = 1.01 * pi
        angle3 = -0.6 * pi
        ptm_ref = np.einsum('ai, ijkl, km -> ajml',
                            ptm_rotate(angle3), ptm_cphase(angle2), ptm_rotate(angle1))

        circuit = grotate(angle=angle1) + gcphase(angle=angle2) + \
                  grotate(angle=angle3)
        assert circuit.ptm(bases2q, bases2q) == approx(ptm_ref)

        circuit = grotate(angle='angle1') + gcphase(angle='angle2') + \
                  grotate(angle='angle3')
        assert circuit(angle1=angle1, angle2=angle2, angle3=angle3)\
                   .ptm(bases2q, bases2q) == approx(ptm_ref)


class TestCircuitsTimeAware:
    def test_gate_create(self):
        dim = 2
        gate = Gate('Q0', dim, lambda: (ptm_rotate(0.5*pi), bases1q, bases1q), 20.)
        assert gate.time_start == 0
        assert gate.time_end == 20.
        assert gate.duration == 20.

        gate = Gate(['Q0', 'Q1'], dim, lambda: (ptm_cphase(0.5*pi), bases2q, bases2q),
                    40., 125.)
        assert gate.time_start == 125.
        assert gate.time_end == approx(165.)
        assert gate.duration == 40.

        gate.time_end = 90.
        assert gate.time_start == approx(50.)
        assert gate.time_end == 90.
        assert gate.duration == 40.

        gate1 = gate.shift(time_start=0.)
        assert gate.time_start == approx(50.)
        assert gate.time_end == 90.
        assert gate.duration == 40.
        assert gate1.time_start == 0.
        assert gate1.time_end == 40.
        assert gate1.duration == 40.

        gate1 = gate.shift(time_end=123.)
        assert gate.time_start == approx(50.)
        assert gate.time_end == 90.
        assert gate.duration == 40.
        assert gate1.time_start == approx(83.)
        assert gate1.time_end == 123.
        assert gate1.duration == 40.

    def test_add_gate_no_init_time(self):
        dim = 2
        gate_q0 = Gate('Q0', dim, lambda: (ptm_rotate(0.5*pi), bases1q, bases1q), 20.)
        gate_2q = Gate(['Q0', 'Q1'], dim,
                       lambda: (ptm_cphase(0.5*pi), bases2q, bases2q), 40.)
        gate_q1 = Gate('Q1', dim, lambda: (ptm_rotate(-0.5*pi), bases1q, bases1q), 30.)

        circuit = gate_q0 + gate_2q
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(60.)
        assert circuit.duration == approx(60.)
        assert [gate.time_start for gate in circuit.gates] == approx([0., 20.])

        circuit = circuit + gate_q1
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(90.)
        assert circuit.duration == approx(90.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 20., 60.])

        circuit = circuit + gate_q0
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(90.)
        assert circuit.duration == approx(90.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 20., 60., 60.])

        circuit = gate_q1 + circuit
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(100.)
        assert circuit.duration == approx(100.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 10., 30., 70., 70.])

        circuit = circuit + (gate_q0 + gate_q1)
        assert circuit.time_start == 0.
        assert circuit.time_end == approx(130.)
        assert circuit.duration == approx(130.)
        assert [gate.time_start for gate in circuit.gates] == \
               approx([0., 10., 30., 70., 70., 100., 100.])

    def test_add_gate_and_delays(self):
        dim = 2

        big_gate = Gate('A', dim, lambda: (ptm_rotate(0.), bases1q, bases1q), 600.)
        rot1 = Gate('D0', dim, lambda: (ptm_rotate(pi), bases1q, bases1q), 20., 290.)
        rot2 = Gate('A', dim, lambda: (ptm_rotate(pi), bases1q, bases1q), 20.)

        # big_gate = Gate('A', dim, lib.rotate_x(0.), 600.)
        # rot1 = Gate('D0', dim, lib.rotate_x(pi), 20., 290.)
        # rot2 = Gate('A', dim, lib.rotate_x(pi), 20., )

        circuit1 = rot2.shift(time_start=20.) + rot2.shift(time_end=120.)
        assert circuit1.time_start == 20.
        assert circuit1.time_end == approx(160.)
        assert circuit1.duration == approx(140.)
        assert [gate.time_start for gate in circuit1.gates] == approx([20., 140.])

        circuit2 = big_gate + rot1
        assert circuit2.time_start == 0.
        assert circuit2.time_end == approx(600.)
        assert circuit2.duration == approx(600.)
        assert [gate.time_start for gate in circuit2.gates] == approx([0., 290.])

        circuit = circuit1 + circuit2
        assert circuit.time_start == 20.
        assert circuit.time_end == approx(760.)
        assert circuit.duration == approx(740.)
        assert [gate.time_start for gate in circuit.gates] == approx(
            [20., 140., 160, 450.])
