import circuit

from unittest.mock import MagicMock, patch, call

import numpy as np


class TestCircuit:
    def test_add_qubit(self):
        c = circuit.Circuit()

        c.add_qubit(circuit.Qubit("A", t1=10, t2=20))

        assert len(c.gates) == 0
        assert len(c.qubits) == 1
        assert c.qubits[0].name == "A"

    def test_order_no_meas(self):
        c = circuit.Circuit()

        qb = circuit.Qubit("A", t1=10, t2=0)
        c.add_qubit(qb)

        c.add_gate(circuit.Hadamard("A", time=1))
        c.add_gate(circuit.Hadamard("A", time=0))

        c.order()

        assert len(c.gates) == 2
        assert c.gates[0].time == 0

    def test_add_waiting_full(self):
        c = circuit.Circuit()

        qb = circuit.Qubit("A", t1=10, t2=0)
        c.add_qubit(qb)

        c.add_gate(circuit.Hadamard("A", time=1))
        c.add_gate(circuit.Hadamard("A", time=0))

        assert len(c.gates) == 2

        c.add_waiting_gates()

        assert len(c.gates) == 3

        c.order()

        assert c.gates[1].time == 0.5
        assert c.gates[1].duration == 1.0

    def test_add_waiting_empty(self):
        c = circuit.Circuit()

        c.add_qubit(circuit.Qubit("A", 0, 0))

        c.add_waiting_gates()

        assert len(c.gates) == 0

        c.add_waiting_gates(tmin=0, tmax=100)

        assert len(c.gates) == 1

    def test_apply_to(self):
        sdm = MagicMock()
        sdm.hadamard = MagicMock()
        sdm.amp_ph_damping = MagicMock()
        sdm.peak_measurement = MagicMock(return_value = (1,0))
        sdm.project_measurement = MagicMock()

        c = circuit.Circuit()

        c.add_qubit(circuit.Qubit("A", 10, np.inf))
        c.add_gate(circuit.Hadamard("A", 0))
        c.add_gate(circuit.Hadamard("A", 10))

        c.add_gate(circuit.Measurement("A", 20, sampler=None))

        c.add_waiting_gates()

        c.order()

        c.apply_to(sdm)

        gamma = 1-np.exp(-1)

        sdm.assert_has_calls([call.hadamard("A"), 
                call.amp_ph_damping("A", gamma=gamma, lamda=0),
                call.hadamard("A"),
                call.amp_ph_damping("A", gamma=gamma, lamda=0),
                call.peak_measurement("A"),
                call.project_measurement("A", 0)])

class TestHadamardGate:
    def test_init(self):
        h = circuit.Hadamard("A", 7)
        assert h.time == 7
        assert h.involves_qubit("A")
        assert not h.involves_qubit("B")
        assert not h.is_measurement

    def test_apply(self):
        sdm = MagicMock()
        sdm.hadamard = MagicMock()
        h = circuit.Hadamard("A", 7)
        h.apply_to(sdm)
        sdm.hadamard.assert_called_once_with("A")

class TestCPhaseGate:
    def test_init(self):
        cp = circuit.CPhase("A", "B", 10)
        assert cp.time == 10
        assert set(cp.involved_qubits) == {"A", "B"}
        assert not cp.is_measurement

    def test_apply(self):
        sdm = MagicMock()
        sdm.cphase = MagicMock()

        cp = circuit.CPhase("A", "B", 10)
        
        cp.apply_to(sdm)

        sdm.cphase.assert_called_once_with("A", "B")

class TestAmpPhDamping:
    def test_init(self):
        apd = circuit.AmpPhDamp("A", 0, 1, 10, 10)
        assert apd.time == 0
        assert apd.duration == 1
        assert apd.involves_qubit("A")
        assert not apd.is_measurement
        

    def test_apply(self):
        sdm = MagicMock()
        sdm.amp_ph_damping = MagicMock()

        apd = circuit.AmpPhDamp("A", 0, 1, 10, 5)
        
        apd.apply_to(sdm)

        g = 1 - np.exp(-1/10)
        l = 1 - np.exp(-1/5)
        sdm.amp_ph_damping.assert_called_once_with("A", gamma=g, lamda=l)

class TestMeasurement:
    def test_init(self):
        m = circuit.Measurement("A", 0, sampler=None)
        assert m.is_measurement
        assert m.involves_qubit("A")
        assert m.time == 0
        
    def test_apply(self):
        m = circuit.Measurement("A", 0, sampler=None)

        sdm = MagicMock()
        sdm.peak_measurement = MagicMock(return_value=(0, 1))
        sdm.project_measurement = MagicMock()

        m.apply_to(sdm)

        assert m.measurements == [1]

        sdm.peak_measurement.assert_called_once_with("A")
        sdm.project_measurement.assert_called_once_with("A", 1)

        sdm.peak_measurement = MagicMock(return_value=(1, 0))
        sdm.project_measurement = MagicMock()

        m.apply_to(sdm)

        assert m.measurements == [1, 0]

        sdm.peak_measurement.assert_called_once_with("A")
        sdm.project_measurement.assert_called_once_with("A", 0)


    def test_apply_random(self):
        m = circuit.Measurement("A", 0, sampler=None)

        with patch('numpy.random.random') as p:
            p.return_value = 0.3
            sdm = MagicMock()
            sdm.peak_measurement = MagicMock(return_value=(0.06, 0.04))
            sdm.project_measurement = MagicMock()

            m.apply_to(sdm)

            sdm.project_measurement.assert_called_once_with("A", 0)

            sdm.peak_measurement = MagicMock(return_value=(0.06, 0.04))
            sdm.project_measurement = MagicMock()
            p.return_value = 0.7

            m.apply_to(sdm)

            sdm.project_measurement.assert_called_once_with("A", 1)


