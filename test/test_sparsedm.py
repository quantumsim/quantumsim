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


def test_ensure_classical_simple():
    sdm = SparseDM(10)
    sdm.ensure_classical(0)


def test_ensure_classical_no_operation():
    sdm = SparseDM(10)
    sdm.ensure_dense(0)
    sdm.ensure_classical(0)


def test_ensure_classical_fail_after_hadamard():
    sdm = SparseDM(10)
    sdm.hadamard(0)
    with pytest.raises(ValueError):
        sdm.ensure_classical(0)


def test_set_bit():
    sdm = SparseDM(10)
    sdm.set_bit(0, 1)

    assert sdm.classical[0] == 1

    sdm.hadamard(0)
    sdm.hadamard(0)


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

    assert np.allclose(p0, 0.02 + 0.98 * 0.02)


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

    assert sdm.last_peak is None
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

    assert sdm.last_peak is None
    assert len(sdm.classical) == 1
    assert 0 in sdm.classical
    assert sdm.classical[0] == 0
    assert len(sdm.idx_in_full_dm) == 0
    assert sdm.full_dm.no_qubits == 0
    assert np.allclose(sdm.trace(), 1)


def test_meas_on_hadamard():
    sdm = SparseDM(1)
    sdm.hadamard(0)

    p0, p1 = sdm.peak_measurement(0)

    assert p0 == 0.5
    assert p1 == 0.5

    sdm.project_measurement(0, 1)

    print(sdm.full_dm.to_array())

    assert len(sdm.classical) == 1
    assert sdm.full_dm.no_qubits == 0
    assert sdm.classical[0] == 1
    assert np.allclose(sdm.trace(), 0.5)


def test_copy():
    sdm = SparseDM(5)
    sdm.classical.update({1: 1, 3: 1})
    sdm.ensure_dense(2)

    assert sdm.full_dm.no_qubits == 1
    assert len(sdm.classical) == 4

    sdm_copy = sdm.copy()

    assert len(sdm_copy.classical) == 4
    assert sdm_copy.classical == {0: 0, 1: 1, 3: 1, 4: 0}
    assert sdm_copy.classical is not sdm.classical
    assert sdm_copy.last_peak is None
    assert sdm.full_dm is not sdm_copy.full_dm
    assert np.allclose(sdm.full_dm.to_array(), sdm_copy.full_dm.to_array())


class TestMultipleMeasurement:

    def test_multiple_measurement_gs(self):
        sdm = SparseDM(3)

        sdm.ensure_dense(0)
        sdm.ensure_dense(1)
        sdm.ensure_dense(2)

        meas = sdm.peak_multiple_measurements([0, 1, 2])

        assert len(meas) == 8
        for state, p in meas:
            if state[0] == 0 and state[1] == 0 and state[2] == 0:
                assert np.allclose(p, 1)
            else:
                assert np.allclose(p, 0)

    def test_multiple_measurement_hadamard_order1(self):
        sdm = SparseDM(3)

        sdm.hadamard(0)
        sdm.hadamard(2)

        sdm.ensure_dense(0)
        sdm.ensure_dense(1)
        sdm.ensure_dense(2)

        meas = sdm.peak_multiple_measurements([0, 1, 2])

        assert len(meas) == 8
        for state, p in meas:
            for x in state.values():
                assert x in [0, 1]
            if state[1] == 0:
                assert np.allclose(p, 0.25)
            else:
                assert np.allclose(p, 0)

    def test_multiple_measurement_hadamard_order2_regression(self):
        sdm = SparseDM(3)

        sdm.hadamard(0)
        sdm.hadamard(1)

        sdm.ensure_dense(0)
        sdm.ensure_dense(1)
        sdm.ensure_dense(2)

        meas = sdm.peak_multiple_measurements([0, 1, 2])

        assert len(meas) == 8
        for state, p in meas:
            print(meas)
            if state[2] == 0:
                assert np.allclose(p, 0.25)
            else:
                assert np.allclose(p, 0)

    def test_multiple_does_not_change(self):
        sdm = SparseDM(3)

        sdm.hadamard(0)
        sdm.hadamard(1)

        sdm.ensure_dense(0)
        sdm.ensure_dense(1)
        sdm.ensure_dense(2)

        before = sdm.full_dm.to_array()

        assert before.shape == (8, 8)
        sdm.peak_multiple_measurements([0, 1, 2])

        assert np.allclose(before, sdm.full_dm.to_array())

    def test_multiple_measurement_hadamard_on_classical(self):
        sdm = SparseDM(2)

        sdm.hadamard(0)

        meas = sdm.peak_multiple_measurements([0, 1])

        assert len(meas) == 2
        assert meas == [({0: 0, 1: 0}, 0.5), ({0: 1, 1: 0}, 0.5)]

    def test_multiple_measurement_only_classical(self):
        sdm = SparseDM(2)

        meas = sdm.peak_multiple_measurements([0])

        assert meas == [({0: 0}, 1)]


def test_renormalize():
    sdm = SparseDM(2)

    sdm.hadamard(0)

    sdm.project_measurement(0, 1)

    assert np.allclose(sdm.trace(), 0.5)

    sdm.renormalize()

    assert np.allclose(sdm.trace(), 1)


def test_rotate_y():
    sdm = SparseDM(2)

    sdm.rotate_y(0, np.pi)
    sdm.rotate_y(1, np.pi)

    assert np.allclose(sdm.trace(), 1)

    sdm.project_measurement(1, 1)
    sdm.project_measurement(0, 1)

    assert np.allclose(sdm.trace(), 1)

    sdm.rotate_y(0, np.pi / 2)
    sdm.rotate_y(1, np.pi / 2)

    sdm.project_measurement(1, 1)
    sdm.project_measurement(0, 1)

    assert np.allclose(sdm.trace(), 0.25)


def test_max_no_bits():
    sdm = SparseDM(3)

    assert sdm.max_bits_in_full_dm == 0

    sdm.ensure_dense(0)
    assert sdm.max_bits_in_full_dm == 1
    sdm.ensure_dense(1)
    assert sdm.max_bits_in_full_dm == 2
    sdm.ensure_dense(2)
    assert sdm.max_bits_in_full_dm == 3

    sdm.ensure_classical(0)
    assert sdm.max_bits_in_full_dm == 3
    sdm.ensure_classical(1)
    assert sdm.max_bits_in_full_dm == 3
    sdm.ensure_classical(2)
    assert sdm.max_bits_in_full_dm == 3

    assert len(sdm.classical) == 3
    assert sdm.full_dm.no_qubits == 0


class TestMajorityVote:
    def test_majority_vote_gs_classical(self):

        bits = [1, 2, 3]
        sdm = SparseDM(bits)

        assert sdm._last_majority_vote_array is None
        assert sdm._last_majority_vote_mask is None

        p = sdm.majority_vote(bits)

        assert sdm._last_majority_vote_mask == 0

        assert np.allclose(p, 0)

    def test_majority_vote_on_excited_classical(self):
        bits = [1, 2, 3]
        sdm = SparseDM(bits)

        sdm.set_bit(1, 1)
        sdm.set_bit(3, 1)

        p = sdm.majority_vote(bits)
        assert np.allclose(p, 1)
        assert sdm._last_majority_vote_mask == 0 

    def test_majority_vote_on_excited_quantum(self):
        bits = [1, 2, 3]
        sdm = SparseDM(bits)

        sdm.rotate_y(1, np.pi)
        sdm.rotate_y(2, np.pi)
        sdm.rotate_y(3, 2*np.pi)

        p = sdm.majority_vote(bits)
        assert np.allclose(p, 1)
        assert sdm._last_majority_vote_mask == 7

    def test_majority_after_hadamard(self):

        bits = [1, 2, 3]
        sdm = SparseDM(bits)
        sdm.hadamard(1)
        sdm.hadamard(2)
        sdm.hadamard(3)

        p = sdm.majority_vote(bits)

        assert np.allclose(p, 0.5)

        sdm.hadamard(3)

        p = sdm.majority_vote(bits)

        assert np.allclose(p, 0.25)

    def test_majority_vote_reuse_of_cached(self):
        #TODO?!
        pass
