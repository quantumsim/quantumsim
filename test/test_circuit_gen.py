import pytest
import numpy as np
from pytest import approx

from quantumsim import Setup, State

# from quantumsim.models import (DiCarloLabQubits, DepolarizingQubits)
# dicarlo_lab_setup = "../quantumsim/setups/two_qubit_DiCarlo.yaml",
# depolarizing_setup = "../quantumsim/setups/two_qubit_depolarizing.yaml"
#test


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize('model_class,setup_file', [
    # [DiCarloLabQubits, dicarlo_lab_setup],
    # [DepolarizingQubits, depolarizing_setup]
])
class TestCircuitGen:
    def test_simple_circuits_gen(self, model_class, setup_file):

        # setup should contain information about two transmon qubits
        # labelled 'q1' and 'q2'
        # names are somewhat arbitrary, not sure how we should set
        # this in the setup file.
        setup = Setup(setup_file)
        model = model_class(setup)

        # The following circuit prepares a Bell state
        # I think the ordering here makes sense?
        gate1 = model.rotate_y('q1', np.pi/2)
        gate2 = model.rotate_y('q2', np.pi/2)
        circuit = gate1 @ gate2

        # Two-qubit gate that generally requires knowledge of which qubit
        # is high-frequency - this should be stored within the yaml file
        # if required.
        gate3 = model.cphase('q1', 'q2', np.pi)

        # I think it makes more sense to use the operator l->r ordering
        # than time-ordering.
        circuit = gate3 @ circuit

        gate4 = model.rotate_y('q1', -np.pi/2)
        circuit = gate4 @ circuit

        circuit.compile()

        qr = Setup(circuit.qubits)
        qr = circuit @ qr

        # Previously there's been all kinds of horrible questions about
        # a) when to apply all pending gates on a circuit
        # b) how to order the resulting output when the dm is called.
        # I suggest that for a), we tag gates as idling or not,
        # and apply
        dm = circuit.rdm(['q1', 'q2'])

        assert dm[0, 0] > 0.5
        assert dm[3, 3] > 0.4
        assert dm[1, 1] < 0.1
        assert dm[2, 2] < 0.1
        assert dm[0, 3] > 0.4
        assert dm[3, 0] > 0.4

    def test_circuit_unfixed_parameters(self, model_class, setup_file):
        # Test that explores a few different ways to
        # set angles in a circuit

        # setup should contain information about two transmon qubits
        # labelled 'q1' and 'q2'
        # names are somewhat arbitrary, not sure how we should set
        # this in the setup file.
        setup = Setup(setup_file)
        model = model_class(setup)

        # So, if I'm not mistaken, the following code runs a little bit
        # against your intention of a gate being static. However, I think
        # that the word 'set' kind of implies that we are modifying the
        # gate object itself. Also, I don't
        gate1 = model.rotate_y('q1')
        gate1.set_angle(np.pi/2)

        # I think possibly the simplest way to let the user
        # parametrize a circuit is to let them set variables to be strings,
        # which gives a variable name that needs to be set in the circuit
        # before compilation.
        gate2 = model.rotate_y('q2', 'theta')  # theta to be set later!

        circuit = gate1 @ gate2

        # I do like the idea of having a generic set function?
        gate3 = model.cphase('q1', 'q2')
        gate3.set(angle=np.pi)

        # I think it makes more sense to use the operator l->r ordering
        # than time-ordering.
        circuit = gate3 @ circuit

        gate4 = model.rotate_y('q1', 'phi')
        gate4.set(phi=-np.pi/2)
        gate4.set(angle=-np.pi/2)
        circuit = gate4 @ circuit

        # To me, the 'compile function' is a promise from the user that
        # the circuit will not have any additional gates added, but
        # some parameters may still change. I'm not sure if this is
        # sufficient for all compilation???
        circuit.compile()

        qr = Setup(circuit.qubits)
        with pytest.raises(NotImplementedError):
            qr = circuit @ qr
        qr = circuit(theta=np.pi/2) @ qr
        dm = qr.rdm(['q1', 'q2'])

        # As __call__ does not edit angle in place, this should *still*
        # raise an error.
        with pytest.raises(NotImplementedError):
            qr = circuit @ qr

        assert dm[0, 0] > 0.5
        assert dm[3, 3] > 0.4
        assert dm[1, 1] < 0.1
        assert dm[2, 2] < 0.1
        assert dm[0, 3] > 0.4
        assert dm[3, 0] > 0.4

        # Now we set the last parameter and the circuit may be
        # applied without qualification
        circuit.set(theta=np.pi/2)
        qr = Setup(circuit.qubits)
        qr = circuit @ qr

        dm = qr.rdm(['q1', 'q2'])

        assert dm[0, 0] > 0.5
        assert dm[3, 3] > 0.4
        assert dm[1, 1] < 0.1
        assert dm[2, 2] < 0.1
        assert dm[0, 3] > 0.4
        assert dm[3, 0] > 0.4

    def test_circuit_qasm_list(self, model_class, setup_file):

        # I dunno so much about parsing strings and
        # especially parsing line breaks, so I'm just
        # going to leave this here.
        qasm_list = [
            "rotate_y q1 " + str(np.pi/2),
            "rotate_y q2 theta",  # sticking with the free angle!
            "cphase q1 q2 " + str(np.pi),
            "rotate_y q1 " + str(-np.pi/2)]

        setup = Setup(setup_file)
        model = model_class(setup)

        circuit = model.parse_qasm(qasm_list).compile()

        # This should be practically the same circuit as previous
        # so we can just use the same set of tests here.
        qr = Setup(circuit.qubits)
        with pytest.raises(NotImplementedError):
            qr = circuit @ qr
        qr = circuit(theta=np.pi/2) @ qr
        dm = qr.rdm(['q1', 'q2'])

        # As __call__ does not edit angle in place, this should *still*
        # raise an error.
        with pytest.raises(NotImplementedError):
            qr = circuit @ qr

        assert dm[0, 0] > 0.5
        assert dm[3, 3] > 0.4
        assert dm[1, 1] < 0.1
        assert dm[2, 2] < 0.1
        assert dm[0, 3] > 0.4
        assert dm[3, 0] > 0.4

        # Now we set the last parameter and the circuit may be
        # applied without qualification
        circuit.set(theta=np.pi/2)
        qr = Setup(circuit.qubits)
        qr = circuit @ qr

        dm = qr.rdm(['q1', 'q2'])

        assert dm[0, 0] > 0.5
        assert dm[3, 3] > 0.4
        assert dm[1, 1] < 0.1
        assert dm[2, 2] < 0.1
        assert dm[0, 3] > 0.4
        assert dm[3, 0] > 0.4


@pytest.mark.skip(reason='Not implemented')
class TestCircuitTimings:
    def test_timings(self):

        # This is a test showing how I'd expect the formatting to work
        # when gates are put together
        setup = Setup(dicarlo_lab_setup)  # DiCarlo qubits will obv. be timed
        model = DiCarloLabQubits(setup)

        # As I've mentioned in the WIP request, I don't think it makes sense to
        # assign a time to a single gate, as we typically treat these timings
        # as relative to other gates in a larger circuit, and I don't see an
        # intuitive way to let the user make some 'times' absolute without
        # unexpected behaviour. Instead, I think that having an 'insert'
        # function should work well.
        circuit = model.rotate_y('q1', np.pi)
        circuit.insert(model.rotate_y('q2', np.pi), time=100)
        with pytest.raises(NotImplementedError):
            # I don't want to let the user add two gates that overlap with
            # each other for what I think are obvious reasons.
            # Assuming that gates are longer than 10ns
            circuit.insert(model.rotate_y('q2', np.pi), time=110)

        circuit.compile()
        qr = Setup(circuit.qubits)
        qr = circuit @ qr
        dm1 = qr.rdm(['q1'])
        dm2 = qr.rdm(['q2'])

        assert dm1[1, 1] > 0.9
        assert dm2[1, 1] > dm1[1, 1]

    def test_timing_failure(self):

        # Quick test that an untimed model fails with the insert command
        setup = Setup(depolarizing_setup)
        model = DepolarizingQubits(setup)
        circuit = model.rotate_y('q1', np.pi)
        with pytest.raises(NotImplementedError):
            circuit.insert(model.rotate_y('q2', np.pi), time=50)

    def test_timing_edits(self):
        # Ok, so let's see. The issue we have here is one of 'what to
        # do when a circuit takes different amount of time on different
        # qubits'. As part of this, we need to note that some circuits might
        # have delay on some qubits at the start of the circuit, but some
        # circuits might have 'fake delay' - i.e. time where qubits should
        # have decayed as part of another circuit.

        setup = Setup(dicarlo_lab_setup)  # DiCarlo qubits will obv. be timed
        model = DiCarloLabQubits(setup)

        # This circuit takes 420ns on q1, but only 220ns on q2
        circuit1 = model.rotate_y('q1', np.pi)
        circuit1.insert(model.rotate_y('q2', np.pi), time=100)
        circuit1 += model.idle('q1', 400)
        circuit1 += model.idle('q2', 100)
        circuit1.compile()

        # This circuit takes 420ns on q1, but 320ns on q2
        circuit1p = model.rotate_y('q1', np.pi)
        circuit1p.insert(model.rotate_y('q2', np.pi), time=100)
        circuit1p += model.idle('q1', 400)
        circuit1p += model.idle('q2', 200)
        circuit1p.compile()

        # This circuit does nothing for the first 100ns on q1.
        # Note that in my opinion this would be different to
        # explicitly adding 100ns of idling time on q1.
        circuit2 = model.rotate_y('q2', np.pi)
        circuit2.insert(model.rotate_y('q1', np.pi), time=100)
        circuit2.compile()

        qr = Setup(circuit1.qubits)
        qr = circuit1 @ qr
        # When circuit2 is tetrised in, q2 should have a 100ns gap
        # and a rest gate should be added here.
        qr = circuit2 @ qr

        qrp = Setup(circuit1p.qubits)
        qrp = circuit1p @ qrp
        # Here, the tetris should work perfectly, and there should be
        # no need for extra resting gates
        qrp = circuit2 @ qrp

        # Not sure exactly how the qubit 'times' should be stored,
        # but anyway (note that these checks don't need the gate time
        # to be 20ns)
        assert qr.times['q1'] == approx(qrp.times['q1'])
        assert qr.times['q1'] == approx(qr.times['q2'])
        assert qr.times['q2'] == approx(qrp.times['q2'])

        assert qr.rdm(['q2']) == approx(qrp.rdm(['q2']))
        assert qr.rdm(['q1']) == approx(qrp.rdm(['q1']))
