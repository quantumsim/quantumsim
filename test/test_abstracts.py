import unittest

from quantumsim import QubitRegister, Circuit, SimpleCircuit
from quantumsim.models import qubits as lib
import numpy as np



class TestQubitRegister(unittest.TestCase):

    def test_init(self):
        qubits = ['q1', 'q2']
        qr = QubitRegister(qubits)
        self.assertEqual(len(qr.qubits), 2)
        self.assertEqual(qr.n_qubits, 2)
        self.assertEqual(qr.state.n_qubits, 2)


class TestCircuit(unittest.TestCase):

    def test_init(self):
        qubits = ['q1', 'q2']
        c = Circuit(qubits)
        self.assertEqual(len(c.qubits), 2)
        qr = QubitRegister(qubits)
        with self.assertRaises(AttributeError):
            c(qr)


class TestSimpleCircuit(unittest.TestCase):

    def test_compile(self):
        qubits = ['q1']
        c = SimpleCircuit(qubits)
        qr = QubitRegister(qubits)
        rotate90 = lib.rotate_x(0.5*np.pi).at(0)
        c.add_gate(rotate90)
        c.add_sequence([rotate90, rotate90])
        self.assertEqual(len(c.gates), 3)
        c.compile()
        c(qr)
