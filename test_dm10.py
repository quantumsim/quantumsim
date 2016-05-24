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
        n = 10
        a = np.zeros((2**n, 2**n))
        dm = dm10.Density(n, a)
        assert dm._block_size == 32
        assert dm._grid_size == 32

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



