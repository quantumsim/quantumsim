import numpy as np
import pycuda.gpuarray as ga
import pytest
import dmcpu as dm10

@pytest.fixture()
def dm():
    return dm10.Density(5)

@pytest.fixture()
def dm_random():
    n = 5
    a = np.random.random((2**n, 2**n))*1j
    a += np.random.random((2**n, 2**n))

    a += a.transpose().conj()
    a = a/np.trace(a)
        
    dm = dm10.Density(n, a)
    return dm


class TestDensityInit:
    def test_ground_state(self):
        dm = dm10.Density(9)

    def test_can_create_empty(self):
        dm = dm10.Density(0)

    def test_dont_make_huge_matrix(self):
        with pytest.raises(ValueError):
            dm10.Density(200)

    def test_numpy_array(self):
        n = 8
        a = np.zeros((2**n, 2**n))
        dm = dm10.Density(n, a)

        assert dm.no_qubits == n

    def test_wrong_data(self):
        with pytest.raises(ValueError):
            dm = dm10.Density(10, "bla")

    def test_wrong_size(self):
        n = 10
        a = np.zeros((2**n, 2**n))
        with pytest.raises(AssertionError):
            dm = dm10.Density(n+1, a)

class TestDensityTrace:
    def test_empty_trace_one(self, dm):
        assert np.allclose(dm.trace(), 1)

    def test_trace_random(self):
        n = 5
        a = np.random.random((2**n, 2**n))*1j
        a += np.random.random((2**n, 2**n))

        # make a hermitian
        a += a.transpose().conj()
        
        dm = dm10.Density(n, a)

        trace_dm = dm.trace()
        trace_np = a.trace()

        assert np.allclose(trace_dm, trace_np)

class TestDensityGetDiag:
    def test_empty_trace_one(self, dm):
        diag = dm.get_diag()
        diag_should = np.zeros(2**5)
        diag_should[0] = 1
        assert np.allclose(diag, diag_should)

    def test_trace_random(self):
        n = 7
        a = np.random.random((2**n, 2**n))*1j
        a += np.random.random((2**n, 2**n))

        # make a hermitian
        a += a.transpose().conj()
        
        dm = dm10.Density(n, a)

        diag_dm = dm.get_diag()
        diag_a = a.diagonal()

        assert np.allclose(diag_dm, diag_a)
        
class TestDensityCPhase:
    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.cphase(10, 11)

    def test_does_nothing_to_ground_state(self, dm):
        a0 = dm.to_array()
        dm.cphase(4, 3)
        a1 = dm.to_array()
        assert np.allclose(a0, a1)
    
    def test_does_something_to_random_state(self, dm_random):
        dm = dm_random
        a0 = dm.to_array()
        dm.cphase(4, 3)
        a1 = dm.to_array()
        assert not np.allclose(a0, a1)

    def test_preserve_trace_empty(self, dm):
        dm.cphase(2, 1)
        assert np.allclose(dm.trace(), 1)
        dm.cphase(2, 3)
        assert np.allclose(dm.trace(), 1)
        dm.cphase(4, 3)
        assert np.allclose(dm.trace(), 1)

    def test_squares_to_one(self, dm):
        a0 = dm.to_array()
        dm.cphase(2, 1)
        dm.cphase(4, 0)
        dm.cphase(2, 1)
        dm.cphase(4, 0)
        a1 = dm.to_array()
        assert np.allclose(a0, a1)

class TestDensityHadamard:
    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.hadamard(10)

    def test_does_something_to_ground_state(self, dm):
        a0 = dm.to_array()
        dm.hadamard(4)
        a1 = dm.to_array()
        assert not np.allclose(a0, a1)

    def test_preserve_trace_ground_state(self, dm):
        dm.hadamard(2)
        assert np.allclose(dm.trace(), 1)
        dm.hadamard(4)
        assert np.allclose(dm.trace(), 1)
        dm.hadamard(0)
        assert np.allclose(dm.trace(), 1)

    def test_squares_to_one(self, dm):
        a0 = dm.to_array()
        dm.hadamard(4)
        dm.hadamard(4)
        a1 = dm.to_array()
        assert np.allclose(a0, a1)

class TestDensityRotateY:
    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.rotate_y(10, 1, 0)

    def test_does_something_to_ground_state(self, dm):
        a0 = dm.to_array()
        dm.rotate_y(4, np.cos(0.5), np.sin(0.5))
        a1 = dm.to_array()
        assert not np.allclose(a0, a1)

    def test_excite(self):
        dm = dm10.Density(2)
        dm.rotate_y(0, np.cos(np.pi/2), np.sin(np.pi/2))
        dm.rotate_y(1, np.cos(np.pi/2), np.sin(np.pi/2))
        
        a1 = dm.to_array()
        assert np.allclose(np.trace(a1), 1)
        assert np.allclose(a1[-1, -1], 1)

class TestDensityAmpPhDamping:
    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.amp_ph_damping(10, 0.0, 0.0)

    def test_does_nothing_to_ground_state(self, dm):
        a0 = dm.to_array()
        dm.amp_ph_damping(4, 0.5, 0.5)
        a1 = dm.to_array()
        assert np.allclose(a0, a1)

    def test_preserve_trace_random_state(self, dm_random):
        dm = dm_random
        dm.amp_ph_damping(4, 0.5, 0.5)
        assert np.allclose(dm.trace(), 1)
        dm.amp_ph_damping(2, 0.5, 0.5)
        assert np.allclose(dm.trace(), 1)
        dm.amp_ph_damping(1, 0.5, 0.5)
        assert np.allclose(dm.trace(), 1)

    def test_strong_damping_gives_ground_state(self, dm_random):
        dm = dm_random
        assert np.allclose(dm.trace(), 1)

        for bit in range(dm.no_qubits):
            dm.amp_ph_damping(bit, 1.0, 0.0)

        assert np.allclose(dm.trace(), 1)

        a2 = dm.to_array()

        assert np.allclose(a2[0, 0], 1)

class TestDensityAddAncilla:
    def test_bit_too_high(self, dm):
        dm = dm10.Density(10)
        with pytest.raises(AssertionError):
            dm.add_ancilla(12, 0)

    def test_add_high_ancilla_to_gs_gives_gs(self):
        dm = dm10.Density(9)
        dm2 = dm.add_ancilla(9, 0)
        assert dm2.no_qubits == 10
        assert np.allclose(dm2.trace(), 1)
        a = dm2.to_array()
        assert np.allclose(a[0, 0], 1)

    def test_add_first_full(self):
        dm = dm10.Density(0)
        dm2 = dm.add_ancilla(0, 0)

        a = dm2.to_array()

        assert np.allclose(a, [[1, 0], [0, 0]])

    def test_add_other_ancilla_to_gs_gives_gs(self, dm):
        dm2 = dm.add_ancilla(4, 0)
        assert dm2.no_qubits == dm.no_qubits + 1
        assert np.allclose(dm2.trace(), 1)
        a = dm2.to_array()
        assert np.allclose(a[0, 0], 1)
        
    def test_add_exc_ancilla_to_gs_gives_no_gs(self, dm):
        dm2 = dm.add_ancilla(4, 1)
        assert dm2.no_qubits == dm.no_qubits + 1
        assert np.allclose(dm2.trace(), 1)
        a = dm2.to_array()
        assert np.allclose(a[0, 0], 0)

    def test_preserve_trace_random_state(self, dm_random):
        dm = dm_random
        assert np.allclose(dm.trace(), 1)
        dm2 = dm.add_ancilla(3, 1)
        assert np.allclose(dm2.trace(), 1)

class TestDensityMeasure:
    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.measure_ancilla(12)

    def test_gs_always_gives_zero(self, dm):
        p0, dm0, p1, dm1 = dm.measure_ancilla(4)

        assert np.allclose(p0, 1)
        assert np.allclose(dm0.to_array()[0, 0], 1)
        assert np.allclose(p1, 0)
        assert np.allclose(dm1.to_array()[0, 0], 0)

    def test_hadamard_gives_50_50(self, dm):
        dm.hadamard(4)
        p0, dm0, p1, dm1 = dm.measure_ancilla(4)

        assert np.allclose(p0, 0.5)
        assert np.allclose(p1, 0.5)

    def test_hadamard_gives_50_50_on_small(self):
        dm = dm10.Density(1)

        dm.hadamard(0)
        p0, dm0, p1, dm1 = dm.measure_ancilla(0)

        assert np.allclose(p0, 0.5)
        assert np.allclose(p1, 0.5)

class TestCopy:
    def test_equality(self, dm):
        dm_copy = dm.copy()
        assert np.allclose(dm.to_array(), dm_copy.to_array())
        
        dm_copy.hadamard(0)
        assert not np.allclose(dm.to_array(), dm_copy.to_array())

class TestRenormalize:
    def test_renormalize_does_nothing_to_gs(self, dm):
        a0 = dm.to_array()
        dm.renormalize()
        a1 = dm.to_array()
        assert np.allclose(a0, a1)

    def test_random_matrix(self):
        n = 6
        a = np.random.random((2**n, 2**n))*1j
        a += np.random.random((2**n, 2**n))
        a += a.transpose().conj()
        
        dm = dm10.Density(n, a)

        dm.renormalize()
        tr = dm.trace()

        a2 = dm.to_array()

        assert np.allclose(tr, 1)
        assert np.allclose(a2, a/np.trace(a))






        



        




