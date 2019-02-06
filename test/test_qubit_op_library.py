# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import pytest
import numpy as np

from qs2.operations.operation import Transformation
from qs2 import bases
from qs2.operations import qubits as lib
from qs2.backends import DensityMatrix


class TestLibrary:
    def test_rotate_x(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis * 3
        dm = DensityMatrix(sys_bases)

        rotate90 = lib.rotate_x(0.5*np.pi)
        rotate180 = lib.rotate_x(np.pi)
        rotate360 = lib.rotate_x(2*np.pi)

        rotate90(dm, 1)
        rotate180(dm, 2)
        assert np.allclose(dm.partial_trace(0), (1, 0))
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))
        assert np.allclose(dm.partial_trace(2), (0, 1))

        rotate180(dm, 1)
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

        rotate90(dm, 1)
        assert np.allclose(dm.partial_trace(1), (1, 0))

        rotate360(dm, 0)
        assert np.allclose(dm.partial_trace(0), (1, 0))

    def test_rotate_y(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis+qubit_basis
        dm = DensityMatrix(sys_bases)

        rotate90 = lib.rotate_y(0.5*np.pi)
        rotate180 = lib.rotate_y(np.pi)
        rotate360 = lib.rotate_y(2*np.pi)

        rotate90(dm, 1)
        rotate180(dm, 2)
        assert np.allclose(dm.partial_trace(0), (1, 0))
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))
        assert np.allclose(dm.partial_trace(2), (0, 1))

        rotate180(dm, 1)
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

        rotate90(dm, 1)
        assert np.allclose(dm.partial_trace(1), (1, 0))

        rotate360(dm, 0)
        assert np.allclose(dm.partial_trace(0), (1, 0))

    def test_rotate_z(self):
        sqrt2 = np.sqrt(2)
        qubit_basis = (bases.general(2),)
        dm = DensityMatrix(qubit_basis)

        rotate90 = lib.rotate_z(0.5*np.pi)
        rotate180 = lib.rotate_z(np.pi)
        rotate360 = lib.rotate_z(2*np.pi)

        rotate90(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])
        rotate180(dm, 0)
        assert np.allclose(dm.expansion(), [1, 0, 0, 0])

        # manually apply a Hadamard gate
        had_expansion = np.array([0.5, 0.5, sqrt2, 0])
        superpos_dm = DensityMatrix(qubit_basis,
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
        dm = DensityMatrix(qubit_basis+qubit_basis)

        rotate90x = lib.rotate_euler(0, 0.5*np.pi, 0)
        rotate90y = lib.rotate_euler(0, 0.5*np.pi, 0)

        rotate90x(dm, 0)
        assert np.allclose(dm.partial_trace(0), (0.5, 0.5))

        rotate90y(dm, 1)
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

    def test_hadamard(self):
        qubit_basis = (bases.general(2),)
        sys_bases = qubit_basis+qubit_basis
        dm = DensityMatrix(sys_bases)

        hadamard = lib.hadamard()

        hadamard(dm, 1)
        assert np.allclose(dm.partial_trace(0), (1, 0))
        assert np.allclose(dm.partial_trace(1), (0.5, 0.5))

        hadamard(dm, 1)
        assert np.allclose(dm.partial_trace(1), (1, 0))


