# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import unittest

from qs2.gates import(
    amplitude_phase_damping_ptm,
    cphase_rotation_ptm,
    rotate_x_ptm,
    rotate_y_ptm,
    rotate_z_ptm)

from qs2.operators import(
    LinearCombPTM,
    ProductPTM,
    TwoKrausPTM)


class TestPTMPrimitives(unittest.TestCase):
    def test_rotate_x(self):
        angle = np.pi/2
        ptm = rotate_x_ptm(angle)
        self.assertEqual(ptm.dim_hilbert,2)
        self.assertAlmostEqual(ptm.op[0,0],np.cos(np.pi/4))
        self.assertAlmostEqual(ptm.op[1,1],np.cos(np.pi/4))
        self.assertAlmostEqual(ptm.op[0,1],-1j*np.sin(np.pi/4))
        self.assertAlmostEqual(ptm.op[1,0],-1j*np.sin(np.pi/4))

    def test_rotate_y(self):
        angle = np.pi/2
        ptm = rotate_y_ptm(angle)
        self.assertEqual(ptm.dim_hilbert,2)
        self.assertAlmostEqual(ptm.op[0,0],np.cos(np.pi/4))
        self.assertAlmostEqual(ptm.op[1,1],np.cos(np.pi/4))
        self.assertAlmostEqual(ptm.op[0,1],-np.sin(np.pi/4))
        self.assertAlmostEqual(ptm.op[1,0],np.sin(np.pi/4))

    def test_rotate_z(self):
        angle = np.pi
        ptm = rotate_z_ptm(angle)
        self.assertEqual(ptm.dim_hilbert,2)
        self.assertAlmostEqual(ptm.op[0,0],-1j)
        self.assertAlmostEqual(ptm.op[1,1],1j)
        self.assertAlmostEqual(ptm.op[0,1],0)
        self.assertAlmostEqual(ptm.op[1,0],0)

    def test_amplitude_phase_damping_type(self):
        gamma = 0.1
        lamda = 0.1
        ptm = amplitude_phase_damping_ptm(gamma, lamda)
        self.assertTrue(isinstance(ptm,ProductPTM))
        self.assertTrue(isinstance(ptm.elements[0],LinearCombPTM))
        self.assertTrue(isinstance(ptm.elements[1],LinearCombPTM))

    def test_cphase_rotation_type(self):
        angle = np.pi
        ptm = cphase_rotation_ptm(angle)
        self.assertTrue(isinstance(ptm,TwoKrausPTM))
