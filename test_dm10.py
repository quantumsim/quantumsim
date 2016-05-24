import dm10
import numpy as np
import pycuda.gpuarray as ga
import pytest

class TestDensityInit:
    def test_empty(self):
        dm = dm10.Density(10)
        assert dm._block_size == 32
        assert dm._grid_size == 32

    def test_dont_make_huge_matrix(self):
        with pytest.raises(ValueError):
            dm10.Density(200)

    def test_numpy_array(self):
        n = 8
        a = np.zeros((2**n, 2**n))
        dm = dm10.Density(n, a)
        assert dm._block_size == 32
        assert dm._grid_size == 8

    def test_gpu_array(self):
        n = 10
        a = ga.zeros((2**n, 2**n), dtype=np.complex128)
        dm = dm10.Density(n, a)
        assert a.gpudata is dm.data.gpudata

    def test_wrong_data(self):
        with pytest.raises(ValueError):
            dm = dm10.Density(10, "bla")

    def test_wrong_size(self):
        n = 10
        a = np.zeros((2**n, 2**n))
        with pytest.raises(AssertionError):
            dm = dm10.Density(n+1, a)


class TestDensityTrace:
    def test_empty_trace_one(self):
        dm = dm10.Density(10)
        assert np.allclose(dm.trace(), 1)

    def test_trace_random(self):
        n = 10
        a = np.random.random((2**n, 2**n))*1j
        a += np.random.random((2**n, 2**n))

        # make a hermitian
        a += a.transpose().conj()
        
        dm = dm10.Density(n, a)

        trace_dm = dm.trace()
        trace_np = a.trace()

        assert np.allclose(trace_dm, trace_np)

        
class TestDensityCPhase:
    def test_bit_too_high(self):
        dm = dm10.Density(10)
        with pytest.raises(AssertionError):
            dm.cphase(10, 11)

    def test_does_nothing_to_ground_state(self):
        dm = dm10.Density(10)
        a0 = dm.data.get()
        dm.cphase(4, 5)
        a1 = dm.data.get()
        assert np.allclose(a0, a1)
    
    def test_does_something_to_random_state(self):
        a = np.random.random((2**10, 2**10))
        dm = dm10.Density(10, a)
        a0 = dm.data.get()
        dm.cphase(4, 5)
        a1 = dm.data.get()
        assert not np.allclose(a0, a1)

    def test_preserve_trace_empty(self):
        dm = dm10.Density(10)
        dm.cphase(2, 8)
        assert np.allclose(dm.trace(), 1)
        dm.cphase(2, 3)
        assert np.allclose(dm.trace(), 1)
        dm.cphase(4, 5)
        assert np.allclose(dm.trace(), 1)

    def test_squares_to_one(self):
        dm = dm10.Density(10)
        a0 = dm.data.get()
        dm.cphase(2, 8)
        dm.cphase(4, 5)
        dm.cphase(2, 8)
        dm.cphase(4, 5)
        a1 = dm.data.get()
        assert np.allclose(a0, a1)



class TestDensityHadamard:
    def test_bit_too_high(self):
        dm = dm10.Density(10)
        with pytest.raises(AssertionError):
            dm.hadamard(10)

    def test_does_something_to_ground_state(self):
        dm = dm10.Density(10)
        a0 = dm.data.get()
        dm.hadamard(4)
        a1 = dm.data.get()
        assert not np.allclose(a0, a1)

    def test_preserve_trace_ground_state(self):
        dm = dm10.Density(10)
        dm.hadamard(2)
        assert np.allclose(dm.trace(), 1)
        dm.hadamard(4)
        assert np.allclose(dm.trace(), 1)
        dm.hadamard(0)
        assert np.allclose(dm.trace(), 1)

    def test_squares_to_one(self):
        dm = dm10.Density(10)
        a0 = dm.data.get()
        dm.hadamard(8)
        dm.hadamard(8)
        a1 = dm.data.get()
        assert np.allclose(a0, a1)


class TestDensityAmpPhDamping:
    def test_bit_too_high(self):
        dm = dm10.Density(10)
        with pytest.raises(AssertionError):
            dm.amp_ph_damping(10, 0.0, 0.0)

    def test_does_nothing_to_ground_state(self):
        dm = dm10.Density(10)
        a0 = dm.data.get()
        dm.amp_ph_damping(4, 0.5, 0.5)
        a1 = dm.data.get()
        assert np.allclose(a0, a1)

    def test_preserve_trace_random_state(self):
        dm = dm10.Density(10)
        dm.amp_ph_damping(4, 0.5, 0.5)
        assert np.allclose(dm.trace(), 1)
        dm.amp_ph_damping(6, 0.5, 0.5)
        assert np.allclose(dm.trace(), 1)
        dm.amp_ph_damping(7, 0.5, 0.5)
        assert np.allclose(dm.trace(), 1)

    def test_strong_damping_gives_ground_state(self):
        n = 5 
        a = np.random.random((2**n, 2**n))*1j
        a += np.random.random((2**n, 2**n))
        # make density matrix
        a = np.dot(a, a.transpose().conj())
        a = a/np.trace(a)
        dm = dm10.Density(n, a)
        assert np.allclose(dm.trace(), 1)

        for bit in range(n):
            dm.amp_ph_damping(bit, 1.0, 0.0)

        assert np.allclose(dm.trace(), 1)

        a2 = dm.data.get()

        assert np.allclose(a2[0, 0], 1)


class TestDensityAddAncilla:
    def test_bit_too_high(self):
        dm = dm10.Density(10)
        with pytest.raises(AssertionError):
            dm.add_ancilla(12, 0)

    def test_add_high_ancilla_to_gs_gives_gs(self):
        dm = dm10.Density(9)
        dm2 = dm.add_ancilla(9, 0)
        assert dm2.no_qubits == 10
        assert np.allclose(dm2.trace(), 1)
        a = dm2.data.get()
        assert np.allclose(a[0, 0], 1)

    def test_add_other_ancilla_to_gs_gives_gs(self):
        dm = dm10.Density(9)
        dm2 = dm.add_ancilla(4, 0)
        assert dm2.no_qubits == 10
        assert np.allclose(dm2.trace(), 1)
        a = dm2.data.get()
        assert np.allclose(a[0, 0], 1)
        
    def test_add_exc_ancilla_to_gs_gives_no_gs(self):
        dm = dm10.Density(9)
        dm2 = dm.add_ancilla(4, 1)
        assert dm2.no_qubits == 10
        assert np.allclose(dm2.trace(), 1)
        a = dm2.data.get()
        assert np.allclose(a[0, 0], 0)

    def test_preserve_trace_random_state(self):
        n = 5 
        a = np.random.random((2**n, 2**n))*1j
        a += np.random.random((2**n, 2**n))
        # make density matrix
        a = np.dot(a, a.transpose().conj())
        a = a/np.trace(a)
        dm = dm10.Density(n, a)
        assert np.allclose(dm.trace(), 1)
        dm2 = dm.add_ancilla(3, 1)
        assert np.allclose(dm2.trace(), 1)

class TestDensityMeasure:
    def test_bit_too_high(self):
        dm = dm10.Density(10)
        with pytest.raises(AssertionError):
            dm.measure_ancilla(12)

    def test_gs_always_gives_zero(self):
        dm = dm10.Density(10)
        p0, dm0, p1, dm1 = dm.measure_ancilla(4)

        assert np.allclose(p0, 1)
        assert np.allclose(dm0.data.get()[0, 0], 1)
        assert np.allclose(p1, 0)
        assert np.allclose(dm1.data.get()[0, 0], 0)


    def test_hadamard_gives_50_50(self):
        dm = dm10.Density(10)
        dm.hadamard(4)
        p0, dm0, p1, dm1 = dm.measure_ancilla(4)

        assert np.allclose(p0, 0.5)
        assert np.allclose(p1, 0.5)

