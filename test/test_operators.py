# This file is part of quantumsim. (https://gitlab.com/quantumsim/quantumsim)
# (c) 2018 Quantumsim Authors
# Distributed under the GNU GPLv3. See LICENSE.txt or
# https://www.gnu.org/licenses/gpl.txt

import numpy as np
import unittest


from qs2.operators import(
    AdjunctionPLM,
    AdjustablePTM,
    AdjustableGate,
    ConjunctionPTM,
    ContainerGate,
    DummyPTM,
    Gate,
    ExplicitBasisPTM,
    LindbladPLM,
    LinearCombPTM,
    PLMIntegrator,
    ProductContainer,
    ProductPTM,
    PTM,
    RotationGate,
    TimedGate,
    TwoKrausPTM,
    TwoPTM,
    TwoPTMProduct,
    TwoPTMExplicit)

from qs2.gates import(
    rotate_x_unitary,
    rotate_y_unitary,
    rotate_z_unitary)


class TestDummyPTM(unittest.TestCase):

    def test_init_raises(self):
        ptm = DummyPTM()
        self.assertFalse(ptm.ready)
        with self.assertRaises(NotImplementedError):
            ptm.get_matrix()


class TestAdjustablePTM(unittest.TestCase):

    def test_adjusts(self):
        def dummy_function():
            return True
        ptm = AdjustablePTM(dummy_function)
        ptm.adjust()
        assert ptm.get_matrix is True


class TestGate(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(NotImplementedError):
            gate = Gate(
                qubits=['test'], setup='test2')
            self.assertEqual(gate.setup,'test2')
            self.assertEqual(gate.qubits[0],'test')
            self.assertEqual(len(gate.qubits),1)
            self.assertEqual(len(gate.ptms), 0)
            self.assertFalse(gate.compiled_flag)

    def test_functions(self):
        with self.assertRaises(NotImplementedError):
            gate = Gate(
                qubits=['test'], setup='test2')
            self.assertEqual(len(gate.get_qubits(),1))
            with self.assertRaises(NotImplementedError):
                gate.compile()
            self.assertTrue(type(gate.requires_from_setup), dict)
            gate.compiled_flag = True
            gate.ptms = 'test'
            self.assertEqual(gate.get_ptms, 'test')


class TestAdjustableGate(unittest.TestCase):
    def test_uargs(self):
        with self.assertRaises(NotImplementedError):
            gate = AdjustableGate(qubits=[], setup=None, test='test')
            self.assertEqual(gate.uargs['test'],'test')


class TestRotationGate(unittest.TestCase):

    def test_init(self):
        def mock_function(arg):
            return arg
        gate = RotationGate(
            qubits=['q0'], setup=None, ptm_function=mock_function)
        self.assertFalse(gate.compiled_flag)
        self.assertEqual(len(gate.qubits), 1)
        self.assertTrue(gate.ptm_function(True))

    def test_compile(self):
        def mock_function(arg):
            return arg
        gate = RotationGate(
            qubits=['q0'], setup=None, ptm_function=mock_function)
        gate.adjust(True)
        gate.compile()
        self.assertTrue(gate.compiled_flag)
        self.assertEqual(gate.ptms[0], True)
        self.assertEqual(len(gate.ptms), 1)

    def test_delayed_compile(self):
        def mock_function(arg):
            return arg

        gate = RotationGate(
            qubits=['q0'], setup=None, ptm_function=mock_function)
        gate.compile()
        self.assertTrue(gate.compiled_flag)
        self.assertEqual(len(gate.ptms), 1)
        self.assertTrue(isinstance(gate.ptms[0], AdjustablePTM))


class TestTimedGate(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(NotImplementedError):
            gate = TimedGate(qubits=[], setup=None,
                start_time=0, end_time=1)
            self.assertEqual(gate.start_time,start_time)
            self.assertEqual(gate.end_time,end_time)


class TestContainerGate(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(NotImplementedError):
            gate = ContainerGate(qubits=[],setup=None)
            self.assertTrue(type(gate.gates) is list)
            self.assertEqual(len(gate.gates), 0)


class TestProductContainer(unittest.TestCase):

    def test_compile(self):
        gate1 = rotate_z_unitary(qubits=['q0'], setup=None, angle=np.pi/2)
        gate2 = rotate_y_unitary(qubits=['q0'], setup=None, angle=np.pi/2)
        big_gate = ProductContainer(
            qubits=['q0'], setup=None, gates=[gate1,gate2])
        big_ptms = big_gate.get_ptms()
        self.assertEqual(len(big_ptms),1)
        self.assertTrue(isinstance(big_ptms[0], TwoPTMProduct))
