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

    assert np.allclose(sdm.trace(), 0.25)

    assert sdm.classical == {'O': 1, 'O2':1}





