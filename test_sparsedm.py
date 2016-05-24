from sparsedm import SparseDM
import numpy as np
import pytest

class TestSparseDMInit:
    def test_init(self):
        sdm = SparseDM(10)
        assert sdm.no_qubits == 10
        assert len(sdm.classical) == 10
        assert sdm.classical[0] == 0




def test_trace():
    sdm = SparseDM(4)
    assert np.allclose(sdm.trace(), 1)


def test_ensure_dense_only_allowed_bits():
    sdm = SparseDM(0)
    with pytest.raises(ValueError):
        sdm.ensure_dense(1)

def test_ensure_dense_simple():
    sdm = SparseDM(10)
    sdm.ensure_dense(0)
    sdm.ensure_dense(1)

    assert len(sdm.classical) == 8
    assert len(sdm.idx_in_full_dm) == 2
    assert sdm.full_dm.no_qubits == 2
    assert np.allclose(sdm.trace(), 1)

def test_cphase_simple():
    sdm = SparseDM(2)
    sdm.cphase(0, 1)
    assert sdm.full_dm.no_qubits == 2

def test_peak_on_ground_state():
    sdm = SparseDM(1)
    sdm.ensure_dense(0)

    p0, p1 = sdm.peak_measurement(0)
    assert p0 == 1
    assert p1 == 0
    assert len(sdm.last_peak) == 3
    assert sdm.last_peak['bit'] == 0

def test_peak_on_hadamard():
    sdm = SparseDM(1)
    sdm.hadamard(0)

    p0, p1 = sdm.peak_measurement(0)

    assert np.allclose(p0, 0.5)
    assert np.allclose(p1, 0.5)


    assert np.allclose(sdm.last_peak[0].trace(), 0.5)
    assert np.allclose(sdm.last_peak[1].trace(), 0.5)


def test_peak_on_decay():
    sdm = SparseDM(1)
    sdm.classical[0] = 1

    p0, p1 = sdm.peak_measurement(0)

    assert np.allclose(p0, 0)
    assert np.allclose(p1, 1)

    sdm.amp_ph_damping(0, 0.02, 0)

    p0, p1 = sdm.peak_measurement(0)

    assert np.allclose(p0, 0.02)
    assert np.allclose(p1, 0.98) 

    sdm.amp_ph_damping(0, 0.02, 0)

    p0, p1 = sdm.peak_measurement(0)

    assert np.allclose(p0, 0.02+0.98*0.02)

def test_peak_then_measure():
    sdm = SparseDM(1)

    assert np.allclose(sdm.trace(), 1)
    sdm.ensure_dense(0)
    assert np.allclose(sdm.trace(), 1)

    p0, p1 = sdm.peak_measurement(0)

    assert np.allclose(p0, 1)
    assert np.allclose(p1, 0)
    assert sdm.last_peak['bit'] == 0

    sdm.project_measurement(0, 0)

    assert sdm.last_peak == None
    assert len(sdm.classical) == 1
    assert 0 in sdm.classical
    assert sdm.classical[0] == 0
    assert len(sdm.idx_in_full_dm) == 0
    assert sdm.full_dm.no_qubits == 0
    assert np.allclose(sdm.trace(), 1)

def test_meas_on_ground_state():
    sdm = SparseDM(1)

    sdm.ensure_dense(0)

    sdm.project_measurement(0, 0)

    assert sdm.last_peak == None
    assert len(sdm.classical) == 1
    assert 0 in sdm.classical
    assert sdm.classical[0] == 0
    assert len(sdm.idx_in_full_dm) == 0
    assert sdm.full_dm.no_qubits == 0
    assert np.allclose(sdm.trace(), 1)


def test_meas_on_hadamard():
    sdm = SparseDM(1)
    sdm.hadamard(0)

    print(sdm.full_dm.data.get())

    p0, p1 = sdm.peak_measurement(0)

    assert p0 == 0.5
    assert p1 == 0.5

    sdm.project_measurement(0, 1)

    print(sdm.full_dm.data.get())

    assert len(sdm.classical) == 1
    assert sdm.full_dm.no_qubits == 0
    assert sdm.classical[0] == 1
    assert np.allclose(sdm.trace(), 0.5)
