# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2020 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import pytest
from numpy import pi
from scipy.linalg import expm

from quantumsim import bases, State
from quantumsim.algebra.tools import random_hermitian_matrix
from quantumsim.models import perfect_qutrits as lib

basis = (bases.general(3),)


# noinspection DuplicatedCode
class TestLibrary:
    def test_rotate_x(self):
        state = State([0, 1, 2], dim=3)

        lib.rotate_x(1, angle=pi/2, foo='bar') @ state
        lib.rotate_x(2, angle=pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0, 0))
        assert np.allclose(state.meas_prob(1), (0.5, 0.5, 0))
        assert np.allclose(state.meas_prob(2), (0, 1, 0))

        lib.rotate_x(1, angle=pi) @ state
        assert np.allclose(state.meas_prob(1), (0.5, 0.5, 0))

        lib.rotate_x(1, angle=pi/2) @ state
        assert np.allclose(state.meas_prob(1), (1, 0, 0))

        lib.rotate_x(0, angle=2*pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0, 0))

    def test_rotate_y(self):
        state = State([0, 1, 2], dim=3)

        lib.rotate_y(1, angle=pi/2, foo='bar') @ state
        lib.rotate_y(2, angle=pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0, 0))
        assert np.allclose(state.meas_prob(1), (0.5, 0.5, 0))
        assert np.allclose(state.meas_prob(2), (0, 1, 0))

        lib.rotate_y(1, angle=pi) @ state
        assert np.allclose(state.meas_prob(1), (0.5, 0.5, 0))

        lib.rotate_y(1, angle=pi/2) @ state
        assert np.allclose(state.meas_prob(1), (1, 0, 0))

        lib.rotate_y(0, angle=2*pi) @ state
        assert np.allclose(state.meas_prob(0), (1, 0, 0))

    def test_rotate_z(self):
        sqrt2 = np.sqrt(2)
        state = State([0], dim=3)

        lib.rotate_z(0, angle=pi/2) @ state
        assert np.allclose(state.to_pv(), [1] + [0] * 8)
        lib.rotate_z(0, angle=pi) @ state
        assert np.allclose(state.to_pv(), [1] + [0] * 8)

        # manually apply a Hadamard gate
        had_expansion = np.array([0.5, 0.5, 0, sqrt2, 0, 0, 0, 0, 0])
        state = State.from_pv([0], had_expansion, basis)

        lib.rotate_z(0, angle=pi) @ state
        assert np.allclose(state.to_pv(),
                           [0.5, 0.5, 0, -sqrt2, 0, 0, 0, 0, 0])

        lib.rotate_z(0, angle=pi/2) @ state
        assert np.allclose(state.to_pv(),
                           [0.5, 0.5, 0, 0, -sqrt2, 0, 0, 0, 0])

        lib.rotate_z(0, angle=pi) @ state
        assert np.allclose(state.to_pv(),
                           [0.5, 0.5, 0, 0, sqrt2, 0, 0, 0, 0])

        lib.rotate_z(0, angle=2*pi) @ state
        assert np.allclose(state.to_pv(),
                           [0.5, 0.5, 0, 0, sqrt2, 0, 0, 0, 0])

    def test_cphase(self):
        rng = np.random.default_rng(seed=999)
        angle = rng.uniform(0, pi)
        generator = np.zeros((9, 9))
        generator[2, 4] = 1
        generator[4, 2] = 1
        unitary = expm(-1j * angle * generator / np.pi)

        dm = random_hermitian_matrix(9, 998)
        dm_res = unitary @ dm @ unitary.conj().T

        state = State.from_dm([0, 1], dm, basis*2)
        lib.cphase(0, 1, angle=angle, foo='bar') @ state
        assert np.allclose(state.to_dm(), dm_res)

    def test_measure(self):
        povm0 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        povm1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        povm2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        dm = random_hermitian_matrix(27, 3)
        state = State.from_dm([0, 1, 2], dm, basis*3)

        povm00 = np.kron(np.kron(povm0, identity), identity)
        lib.measure(0, result=0, foo='bar') @ state
        dm = povm00 @ dm @ povm00
        assert np.allclose(state.to_dm(), dm)
        lib.measure(0, result=0) @ state
        assert np.allclose(state.to_dm(), dm)

        povm11 = np.kron(np.kron(identity, povm1), identity)
        lib.measure(1, result=1) @ state
        dm = povm11 @ dm @ povm11
        assert np.allclose(state.to_dm(), dm)

        povm22 = np.kron(np.kron(identity, identity), povm2)
        lib.measure(2, result=2) @ state
        dm = povm22 @ dm @ povm22
        assert np.allclose(state.to_dm(), dm)

        lib.measure(1, result=0) @ state
        assert np.allclose(state.to_dm(), np.zeros((27, 27)))

        with pytest.raises(ValueError, match='Unknown measurement result: 3'):
            lib.measure(0, result=3)

    def test_dephase(self):
        dm = random_hermitian_matrix(3, 3)
        state = State.from_dm([0], dm, basis)
        lib.dephase(0, foo='bar') @ state
        assert np.allclose(state.to_dm(), np.diag(np.diag(dm)))
