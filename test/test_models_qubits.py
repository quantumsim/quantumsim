# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np

from qs2.operations.operation import Operation
from qs2 import bases
from qs2.models import qubits as lib
from qs2.states import State


class TestLibrary:
    def test_rotate_x(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis * 3
        dm = State(sys_bases)

        rotate90 = lib.rotate_x(0.5*np.pi)
        rotate180 = lib.rotate_x(np.pi)
        rotate360 = lib.rotate_x(2*np.pi)

        rotate90(dm, 1)
        rotate180(dm, 2)
        assert np.allclose(dm.meas_prob(0), (1, 0))
        assert np.allclose(dm.meas_prob(1), (0.5, 0.5))
        assert np.allclose(dm.meas_prob(2), (0, 1))

        rotate180(dm, 1)
        assert np.allclose(dm.meas_prob(1), (0.5, 0.5))

        rotate90(dm, 1)
        assert np.allclose(dm.meas_prob(1), (1, 0))

        rotate360(dm, 0)
        assert np.allclose(dm.meas_prob(0), (1, 0))

    def test_rotate_y(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis+qubit_basis
        dm = State(sys_bases)

        rotate90 = lib.rotate_y(0.5*np.pi)
        rotate180 = lib.rotate_y(np.pi)
        rotate360 = lib.rotate_y(2*np.pi)

        rotate90(dm, 1)
        rotate180(dm, 2)
        assert np.allclose(dm.meas_prob(0), (1, 0))
        assert np.allclose(dm.meas_prob(1), (0.5, 0.5))
        assert np.allclose(dm.meas_prob(2), (0, 1))

        rotate180(dm, 1)
        assert np.allclose(dm.meas_prob(1), (0.5, 0.5))

        rotate90(dm, 1)
        assert np.allclose(dm.meas_prob(1), (1, 0))

        rotate360(dm, 0)
        assert np.allclose(dm.meas_prob(0), (1, 0))

    def test_rotate_z(self):
        sqrt2 = np.sqrt(2)
        qubit_basis = (bases.general(2),)
        dm = State(qubit_basis)

        rotate90 = lib.rotate_z(0.5*np.pi)
        rotate180 = lib.rotate_z(np.pi)
        rotate360 = lib.rotate_z(2*np.pi)

        rotate90(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])
        rotate180(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])

        # manually apply a Hadamard gate
        had_expansion = np.array([0.5, 0.5, sqrt2, 0])
        superpos_dm = State(qubit_basis,
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
        dm = State(qubit_basis + qubit_basis)

        rotate90x = lib.rotate_euler(0, 0.5*np.pi, 0)
        rotate90y = lib.rotate_euler(0, 0.5*np.pi, 0)

        rotate90x(dm, 0)
        assert np.allclose(dm.meas_prob(0), (0.5, 0.5))

        rotate90y(dm, 1)
        assert np.allclose(dm.meas_prob(1), (0.5, 0.5))

    def test_hadamard(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis
        dm = State(sys_bases)

        hadamard = lib.hadamard()

        hadamard(dm, 1)
        assert np.allclose(dm.meas_prob(0), (1, 0))
        assert np.allclose(dm.meas_prob(1), (0.5, 0.5))

        hadamard(dm, 1)
        assert np.allclose(dm.meas_prob(1), (1, 0))

    def test_cnot(self):
        import qs2.models.qubits as lib
        cnot = lib.cnot()
        qubit_bases = (bases.general(2),
                       bases.general(2),
                       bases.general(2))

        dm = np.diag([0.25, 0, 0.75, 0, 0, 0, 0, 0])
        s = State.from_dm(qubit_bases, dm)
        assert np.allclose(s.meas_prob(0), (1, 0))
        assert np.allclose(s.meas_prob(1), (0.25, 0.75))
        assert np.allclose(s.meas_prob(2), (1, 0))
        cnot(s, 0, 1)
        assert np.allclose(s.meas_prob(0), (1, 0))
        assert np.allclose(s.meas_prob(1), (0.25, 0.75))
        assert np.allclose(s.meas_prob(2), (1, 0))
        cnot(s, 1, 2)
        assert np.allclose(s.meas_prob(0), (1, 0))
        assert np.allclose(s.meas_prob(1), (0.25, 0.75))
        assert np.allclose(s.meas_prob(2), (0.25, 0.75))

        #
        # dm = np.zeros((4, 4), dtype=complex)
        # dm[0, 0] = 1.
        # s = State.from_dm(qubit_bases, dm)
        # assert np.allclose(s.meas_prob(0), (1, 0))
        # assert np.allclose(s.meas_prob(1), (1, 0))
        # cnot(s, 0, 1)
        # assert np.allclose(s.meas_prob(0), (1, 0))
        # assert np.allclose(s.meas_prob(1), (1, 0))
