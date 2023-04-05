# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import pytest
from numpy import pi

from quantumsim import bases, StateNumpy as State
from quantumsim.algebra.tools import random_hermitian_matrix
from quantumsim.models import perfect_qubits as lib

basis = (bases.general(2),)
basis2 = basis * 2


# noinspection DuplicatedCode
class TestLibrary:
    def test_rotate_x(self):
        state = State(list(range(3)))

        lib.rotate_x(1, angle=0.5 * np.pi, foo="bar") @ state
        lib.rotate_x(2, angle=pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0))
        assert np.allclose(state.meas_prob(1), (0.5, 0.5))
        assert np.allclose(state.meas_prob(2), (0, 1))

        lib.rotate_x(1, angle=pi) @ state
        assert np.allclose(state.meas_prob(1), (0.5, 0.5))

        lib.rotate_x(1, angle=0.5 * pi) @ state
        assert np.allclose(state.meas_prob(1), (1, 0))

        lib.rotate_x(0, angle=2 * pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0))

    def test_rotate_y(self):
        state = State(list(range(3)))

        lib.rotate_y(1, angle=0.5 * pi, foo="bar") @ state
        lib.rotate_y(2, angle=pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0))
        assert np.allclose(state.meas_prob(1), (0.5, 0.5))
        assert np.allclose(state.meas_prob(2), (0, 1))

        lib.rotate_y(1, angle=pi) @ state
        assert np.allclose(state.meas_prob(1), (0.5, 0.5))

        lib.rotate_y(1, angle=0.5 * pi) @ state
        assert np.allclose(state.meas_prob(1), (1, 0))

        lib.rotate_y(0, angle=2 * pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0))

    def test_rotate_z(self):
        sqrt2 = np.sqrt(2)
        state = State([0])

        lib.rotate_z(0, angle=0.5 * pi, foo="bar") @ state
        assert np.allclose(state.to_pv(), [1, 0, 0, 0])
        lib.rotate_z(0, angle=pi) @ state
        assert np.allclose(state.to_pv(), [1, 0, 0, 0])

        # manually apply a Hadamard gate
        had_expansion = np.array([0.5, 0.5, sqrt2, 0])
        state = state.from_pv(had_expansion, basis)

        lib.rotate_z(0, angle=pi) @ state
        assert np.allclose(state.to_pv(), [0.5, 0.5, -sqrt2, 0])

        lib.rotate_z(0, angle=0.5 * pi) @ state
        assert np.allclose(state.to_pv(), [0.5, 0.5, 0, -sqrt2])

        lib.rotate_z(0, angle=pi) @ state
        assert np.allclose(state.to_pv(), [0.5, 0.5, 0, sqrt2])

        lib.rotate_z(0, angle=2 * pi) @ state
        assert np.allclose(state.to_pv(), [0.5, 0.5, 0, sqrt2])

    def test_rotate_euler(self):
        state = State([0, 1])

        lib.rotate_euler(0, angle_z1=0, angle_x=0.5 * pi, angle_z2=0, foo="bar") @ state
        assert np.allclose(state.meas_prob(0), (0.5, 0.5))

        lib.rotate_euler(
            1, angle_z1=0.5 * pi, angle_x=0.5 * pi, angle_z2=-0.5 * pi
        ) @ state
        assert np.allclose(state.meas_prob(1), (0.5, 0.5))

    def test_hadamard(self):
        state = State([0, 1])

        lib.hadamard(1, foo="bar") @ state
        assert np.allclose(state.meas_prob(0), (1, 0))
        assert np.allclose(state.meas_prob(1), (0.5, 0.5))

        lib.hadamard(1) @ state
        assert np.allclose(state.meas_prob(1), (1, 0))

    def test_cphase(self):
        state = State([0, 1])
        lib.cphase(0, 1, angle=pi) @ state
        assert np.allclose(state.diagonal(), [1, 0, 0, 0])

        state = State.from_dm(
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]]),
            basis2,
        )
        lib.cphase(0, 1, angle=0.5 * pi) @ state
        assert np.allclose(
            state.to_dm(),
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, -0.5j], [0, 0, 0.5j, 0.5]],
        )

    def test_cnot(self):
        state = State.from_dm(np.diag([0.25, 0, 0.75, 0, 0, 0, 0, 0]), basis * 3)
        assert np.allclose(state.meas_prob(0), (1, 0))
        assert np.allclose(state.meas_prob(1), (0.25, 0.75))
        assert np.allclose(state.meas_prob(2), (1, 0))
        lib.cnot(0, 1, foo="bar") @ state
        assert np.allclose(state.meas_prob(0), (1, 0))
        assert np.allclose(state.meas_prob(1), (0.25, 0.75))
        assert np.allclose(state.meas_prob(2), (1, 0))
        lib.cnot(1, 2) @ state
        assert np.allclose(state.meas_prob(0), (1, 0))
        assert np.allclose(state.meas_prob(1), (0.25, 0.75))
        assert np.allclose(state.meas_prob(2), (0.25, 0.75))

    def test_swap(self):
        dm = random_hermitian_matrix(4, 3)
        unitary = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        dm_res = unitary @ dm @ unitary.conj().T

        state = State.from_dm(dm, basis2)
        lib.swap(0, 1, foo="bar") @ state
        assert np.allclose(state.to_dm(), dm_res)

    def test_measure(self):
        povm0 = np.array([[1, 0], [0, 0]])
        povm1 = np.array([[0, 0], [0, 1]])
        identity = np.array([[1, 0], [0, 1]])

        dm = random_hermitian_matrix(4, 3)
        state = State.from_dm(dm, basis2)

        povm00 = np.kron(povm0, identity)
        lib.measure(0, result=0, foo="bar") @ state
        dm = povm00 @ dm @ povm00
        assert np.allclose(state.to_dm(), dm)
        lib.measure(0, result=0) @ state
        assert np.allclose(state.to_dm(), dm)

        povm11 = np.kron(identity, povm1)
        lib.measure(1, result=1) @ state
        dm = povm11 @ dm @ povm11
        assert np.allclose(state.to_dm(), dm)

        lib.measure(1, result=0) @ state
        assert np.allclose(state.to_dm(), np.zeros((4, 4)))

        with pytest.raises(ValueError, match="Unknown measurement result: 2"):
            lib.measure(0, result=2)

    def test_dephase(self):
        dm = random_hermitian_matrix(2, 777)
        state = State.from_dm(dm, basis)
        lib.dephase(0, foo="bar") @ state
        assert np.allclose(state.to_dm(), np.diag(np.diag(dm)))

    def test_reset(self):
        dm = random_hermitian_matrix(2**3, 876)
        state = State.from_dm(dm, basis * 3)
        qubit = 0
        lib.reset(qubit, bra="ket") @ state
        assert state.meas_prob(qubit) == pytest.approx([1, 0])
