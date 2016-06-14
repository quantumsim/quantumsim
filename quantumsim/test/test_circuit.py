import quantumsim.circuit as circuit
from unittest.mock import MagicMock, patch, call
import numpy as np




class TestCircuit:
    def test_add_qubit(self):
        c = circuit.Circuit()

        c.add_qubit(circuit.Qubit("A", t1=10, t2=20))

        assert len(c.gates) == 0
        assert len(c.qubits) == 1
        assert c.qubits[0].name == "A"

    def test_get_qubit_names(self):
        c = circuit.Circuit()
        c.add_qubit("A")
        c.add_qubit("B")
        c.add_qubit("C")

        assert set(c.get_qubit_names()) == {"A", "B", "C"}

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

    def test_add_waiting_partial(self):
        c = circuit.Circuit()
        c.add_qubit("A", 10, 10)
        c.add_qubit("B", 10, 10)

        assert len(c.gates) == 0

        c.add_waiting_gates(only_qubits=["A"], tmin=0, tmax=1)

        assert len(c.gates) == 1
        
        c.add_waiting_gates(only_qubits=["B"], tmin=0, tmax=1)

        assert len(c.gates) == 2

    def test_add_waiting_not_to_inf_qubits(self):
        c = circuit.Circuit()
        c.add_qubit("A") #should have infinite lifetime by default
        c.add_qubit("B", np.inf, np.inf) #should have infinite lifetime by default
        c.add_qubit("C", 10, 10)

        c.add_waiting_gates(tmin=0, tmax=1)

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

    def test_simplified_adding_qubit(self):
        c = circuit.Circuit()
        c.add_qubit("A", 10, 10)

        assert len(c.qubits) == 1
        assert c.qubits[0].name == 'A'

    def test_add_gate_by_class(self):
        c = circuit.Circuit()
        c.add_qubit("A", 10, 10)
        
        c.add_gate(circuit.Hadamard, "A", time=20)

        assert len(c.gates) == 1
        assert c.gates[0].time == 20

    def test_add_gate_by_class_name(self):
        c = circuit.Circuit()
        c.add_qubit("A", 10, 10)
        
        c.add_gate("hadamard", "A", time=20)

        assert len(c.gates) == 1
        assert c.gates[0].time == 20

        c.add_gate("cphase", "A", "B", time=30)

        assert len(c.gates) == 2
        assert c.gates[-1].time == 30

    def test_add_gate_by_getattr(self):
        c = circuit.Circuit()
        c.add_qubit("A", 10, 10)
        
        c.add_hadamard("A", time=20)
        c.add_rotate_y("A", time=20, angle=0)

        assert len(c.gates) == 2
        assert c.gates[0].time == 20

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

class TestRotateYGate:
    def test_init(self):
        h = circuit.RotateY("A", 7, 0)
        assert h.time == 7
        assert h.involves_qubit("A")
        assert not h.involves_qubit("B")
        assert not h.is_measurement

        assert h.label == r"$R_y(0)$"

    def test_label_pi(self):
        h = circuit.RotateY("A", 7, np.pi)
        assert h.label == r"$R_y(\pi)$"

    def test_label_piover2(self):
        h = circuit.RotateY("A", 7, np.pi/2)
        assert h.label == r"$R_y(\pi/2)$"

    def test_apply_piover2(self):
        sdm = MagicMock()
        sdm.rotate_y = MagicMock()
        h = circuit.RotateY("A", 7, np.pi/2)
        h.apply_to(sdm)
        sdm.rotate_y.assert_called_once_with("A", angle=np.pi/2)

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
        
    def test_apply_with_uniform_sampler(self):
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

    def test_apply_with_selection_sampler(self):
        m = circuit.Measurement("A", 0, sampler=circuit.selection_sampler(1))

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

        assert m.measurements == [1, 1]

        sdm.peak_measurement.assert_called_once_with("A")
        sdm.project_measurement.assert_called_once_with("A", 1)

    def test_apply_random(self):
        m = circuit.Measurement("A", 0, sampler=None)

        with patch('numpy.random.RandomState') as rsclass:
            rs = MagicMock()
            rsclass.return_value = rs
            rs.random_sample = MagicMock()
            p = rs.random_sample
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

    def test_output_bit(self):
        m = circuit.Measurement("A", 0, sampler=None, output_bit="O")

        sdm = MagicMock()
        sdm.peak_measurement = MagicMock(return_value=(0, 1))
        sdm.project_measurement = MagicMock()
        sdm.set_bit = MagicMock()

        m.apply_to(sdm)

        assert m.measurements == [1]

        sdm.peak_measurement.assert_called_once_with("A")
        sdm.project_measurement.assert_called_once_with("A", 1)
        sdm.set_bit.assert_called_once_with("O", 1)

        sdm.peak_measurement = MagicMock(return_value=(1, 0))
        sdm.project_measurement = MagicMock()
        sdm.set_bit = MagicMock()

        m.apply_to(sdm)

        assert m.measurements == [1, 0]

        sdm.peak_measurement.assert_called_once_with("A")
        sdm.project_measurement.assert_called_once_with("A", 0)
        sdm.set_bit.assert_called_once_with("O", 0)

class TestConditionalGates:
    def test_simple(self):
        sdm = MagicMock()
        sdm.classical = {"A": 0, "B": 1}
        sdm.hadamard = MagicMock()
        

        c = circuit.Circuit()

        c.add_gate("hadamard", "A", time=0, conditional_bit="B")

        c.apply_to(sdm)

        sdm.hadamard.assert_called_once_with("A")
        sdm.ensure_classical.assert_called_once_with("B")

        sdm = MagicMock()
        sdm.classical = {"A": 0, "B": 0}
        sdm.hadamard = MagicMock()

        c.apply_to(sdm)

        sdm.hadamard.assert_not_called()
        sdm.ensure_classical.assert_called_once_with("B")

class TestSamplers:
    def test_selection_sampler(self):
        s = circuit.selection_sampler(0)
        next(s)
        for _ in range(10):
            pr, dec, prob = s.send((0.5, 0.5))
            assert pr == 0
            assert dec == 0
            assert prob == 1

    def test_uniform_sampler(self):
        s = circuit.uniform_sampler()
        with patch("numpy.random.RandomState") as rsclass:
            rs = MagicMock()
            rsclass.return_value = rs
            rs.random_sample = MagicMock(return_value = 0.5)

            s = circuit.uniform_sampler(seed=42)
            next(s)

            rsclass.assert_called_once_with(42)

            for p0 in np.linspace(0, 1, 10):
                proj, dec, prob = s.send((p0, 1 - p0))
                assert proj == dec
                assert prob == 1
                assert proj == int(p0 < 0.5)

    def test_uniform_noisy_sampler(self):
        s = circuit.uniform_sampler()
        with patch("numpy.random.RandomState") as rsclass:
            rs = MagicMock()
            rsclass.return_value = rs
            rs.random_sample = MagicMock(return_value = 0.5)

            s = circuit.uniform_noisy_sampler(0.4, seed=42)
            next(s)

            rsclass.assert_called_once_with(42)

            # no readout error
            dec, proj, prob = s.send((0.2, 0.8))
            assert (proj, dec, prob) == (1, 1, 0.6)
            dec, proj, prob = s.send((0.9, 0.1))
            assert (proj, dec, prob) == (0, 0, 0.6)

            s = circuit.uniform_noisy_sampler(0.7, seed=42)
            next(s)
            dec, proj, prob = s.send((0.2, 0.8))
            assert (proj, dec, prob) == (1, 0, 0.7)
            dec, proj, prob = s.send((0.9, 0.1))
            assert (proj, dec, prob) == (0, 1, 0.7)
