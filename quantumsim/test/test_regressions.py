import quantumsim as qs
import numpy as np


def test_regression_can_change_ptms():
    c = qs.circuit.Circuit()
    c.add_qubit("A")
    gate = c.add_gate(qs.circuit.RotateX("A", time=10, angle=np.pi/2))

    sdm = qs.sparsedm.SparseDM(c.get_qubit_names())

    c.apply_to(sdm)

    assert np.allclose(sdm.full_dm.get_diag(), [0.5, 0.5])

    gate.ptm = qs.ptm.rotate_x_ptm(angle=-np.pi/2)

    c.apply_to(sdm)

    assert np.allclose(sdm.full_dm.get_diag(), [1, 0])

    

