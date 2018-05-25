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

def test_regression_toarray():
    state = qs.sparsedm.SparseDM(9)

    for i in range(1, 9):
        state.ensure_dense(i)

    state.ensure_dense(0)

    g = qs.ptm.rotate_x_ptm(np.pi)

    for i in range(1, 4):
        state.apply_ptm(i, g)

    state.apply_all_pending()

    d1 = state.full_dm.get_diag()

    d2 = np.diag(state.full_dm.to_array())

    assert np.allclose(d1, d2)
