import circuit
import sparsedm
import numpy as np


def test_three_qbit_clean():
    c = circuit.Circuit()

    qubit_names = ["D1", "A1", "D2", "A2", "D3"]

    for qb in qubit_names:
        c.add_qubit(qb, np.inf, np.inf)
    
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


    



