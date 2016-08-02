import quantumsim.circuit as circuit
import quantumsim.sparsedm as sparsedm
import numpy as np

import pytest




def test_three_qbit_clean():
    c = circuit.Circuit()

    qubit_names = ["D1", "A1", "D2", "A2", "D3"]

    # clean ancillas have infinite life-time
    for qb in qubit_names:
        #set lifetime to only almost inf so that waiting gates are added but ineffective
        c.add_qubit(qb, np.inf, 1e10) 
    
    c.add_hadamard("A1", time=0)
    c.add_hadamard("A2", time=0)

    c.add_cphase("A1", "D1", time=200)
    c.add_cphase("A2", "D2", time=200)

    c.add_cphase("A1", "D2", time=100)
    c.add_cphase("A2", "D3", time=100)

    c.add_hadamard("A1", time=300)
    c.add_hadamard("A2", time=300)

    m1 = circuit.Measurement("A1", time=350, sampler=None)
    c.add_gate(m1)
    m2 = circuit.Measurement("A2", time=350, sampler=None)
    c.add_gate(m2)

    c.add_waiting_gates(tmin=0, tmax=1500)

    c.order()

    assert len(c.gates) == 27

    sdm = sparsedm.SparseDM(qubit_names)

    for bit in sdm.classical:
        sdm.classical[bit] = 1

    sdm.classical["D3"] = 0

    assert sdm.classical == {'A1': 1, 'A2': 1, 'D3': 0, 'D1': 1, 'D2': 1}

    for i in range(100):
        c.apply_to(sdm)

    sdm.apply_all_pending()

    assert len(m1.measurements) == 100
    assert len(m2.measurements) == 100

    assert sdm.classical == {}

    #in a clean run, we expect just one possible path
    assert np.allclose(sdm.trace(), 1)

    assert m1.measurements == [1]*100
    assert m2.measurements == [0, 1]*50

def test_noisy_measurement_sampler():
    c = circuit.Circuit()
    c.add_qubit("A", 0, 0)


    c.add_hadamard("A", 1)

    sampler = circuit.uniform_noisy_sampler(seed=42, readout_error = 0.1)
    m1 = c.add_measurement("A", time=2, sampler=sampler)

    sdm = sparsedm.SparseDM("A")

    true_state = []
    for _ in range(20):
        c.apply_to(sdm)
        true_state.append(sdm.classical['A'])

    # these samples assume a certain seed (=42)
    assert m1.measurements == [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1]
    assert true_state != m1.measurements
    assert true_state == [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]

    # we have two measurement errors
    mprob = 0.9**18 * 0.1**2
    assert np.allclose(sdm.classical_probability, mprob)
    
    # and each measurement has outcome 1/2
    totprob = mprob * 0.5**20
    assert np.allclose(sdm.trace(), totprob)

def test_measurement_with_output_bit():
    c = circuit.Circuit()
    c.add_qubit("A")

    c.add_qubit("O")
    c.add_qubit("O2")

    c.add_rotate_y("A", time=0, angle=np.pi/2)

    sampler = circuit.selection_sampler(1)
    c.add_measurement("A", time=1, sampler=sampler, output_bit="O")

    c.add_rotate_y("A", time=3.5, angle=np.pi/2)

    sampler = circuit.selection_sampler(1)
    c.add_measurement("A", time=4, sampler=sampler, output_bit="O2")

    c.add_rotate_y("A", time=5, angle=np.pi/2)
    c.order()

    sdm = sparsedm.SparseDM(c.get_qubit_names())

    assert sdm.classical['O'] == 0
    assert sdm.classical['O2'] == 0


    c.apply_to(sdm)

    sdm.apply_all_pending()

    assert np.allclose(sdm.trace(), 0.25)

    assert sdm.classical == {'O': 1, 'O2':1}

@pytest.mark.skip()
def test_integration_surface17():
    def make_circuit(t1=np.inf, t2=np.inf, seed=42, readout_error=0.015, t_gate=40, t_rest=1000):
        surf17 = circuit.Circuit("Surface 17")

        t_rest += t_gate  # nominal rest time is between two gates

        x_bits = ["X%d" % i for i in range(4)]
        z_bits = ["Z%d" % i for i in range(4)]

        d_bits = ["D%d" % i for i in range(9)]

        for b in x_bits + z_bits + d_bits:
            surf17.add_qubit(b, t1, t2)

        def add_x(c, x_anc, d_bits, t=0, t_gate=t_gate):
            t += t_gate
            for d in d_bits:
                if d is not None:
                    c.add_cphase(d, x_anc, time=t)
                t += t_gate

        add_x(surf17, "X0", [None, None, "D2", "D1"], t=0)
        add_x(surf17, "X1", ["D1", "D0", "D4", "D3"], t=0)
        add_x(surf17, "X2", ["D5", "D4", "D8", "D7"], t=0)
        add_x(surf17, "X3", ["D7", "D6", None, None], t=0)

        t2 = 4 * t_gate + t_rest

        add_x(surf17, "Z0", ["D0", "D3", None, None], t=t2)
        add_x(surf17, "Z1", ["D2", "D5", "D1", "D4"], t=t2)
        add_x(surf17, "Z2", ["D4", "D7", "D3", "D6"], t=t2)
        add_x(surf17, "Z3", [None, None, "D5", "D8"], t=t2)

        sampler = circuit.BiasedSampler(
            readout_error=readout_error, alpha=1, seed=seed)

        for b in x_bits + d_bits:
            surf17.add_hadamard(b, time=0)
            surf17.add_hadamard(b, time=5 * t_gate)

        for b in z_bits:
            surf17.add_hadamard(b, time=4 * t_gate + t_rest)
            surf17.add_hadamard(b, time=4 * t_gate + t_rest + 5 * t_gate)

        for b in z_bits:
            surf17.add_measurement(b, time=10 * t_gate + t_rest, sampler=sampler)

        for b in x_bits:
            surf17.add_measurement(b, time=6 * t_gate, sampler=sampler)

        surf17.add_waiting_gates(
            only_qubits=x_bits, tmax=6 * t_gate, tmin=-t_rest - 5 * t_gate)
        surf17.add_waiting_gates(only_qubits=z_bits + d_bits, tmin=0)

        surf17.order()

        return surf17

    def syndrome_to_byte(syndrome):
        byte = 0

        for i in range(4):
            byte += syndrome["X%d"%i] << (i+4)
        for i in range(4):
            byte += syndrome["Z%d"%i] << i 

        return byte


    seed = 890793515

    t1 = 25000.0
    t2 = 35000.0
    ro_error = 0.015
    t_gate = 40.0
    t_rest = 1000.0

    rounds = 20

    c = make_circuit(t1=t1, t2=t2, seed=seed,
                     readout_error=ro_error, t_gate=t_gate, t_rest=t_rest)


    sdm = sparsedm.SparseDM(c.get_qubit_names())
    for b in ["D%d"%i for i in range(9)]:
        sdm.ensure_dense(b)


    syndromes = []
    for _ in range(rounds):
        c.apply_to(sdm)

        sdm.renormalize()

        syndromes.append(syndrome_to_byte(sdm.classical))

    syndrome = bytes(syndromes)

    assert syndrome == b'jHhJhL\x08L\tK)K\x08K\x08K\x08K\x08I'

