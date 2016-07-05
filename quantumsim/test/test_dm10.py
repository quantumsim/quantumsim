import numpy as np
import pytest

import quantumsim.dmcpu as dmcpu

# There are two implementations for the backend (on CPU and on GPU)
# here we collect the classes we want to test
implementations_to_test = []

#implementations_to_test.append(dmcpu.Density)

hascuda = False
try:
    import pycuda.gpuarray as ga
    import quantumsim.dm10 as dm10
    implementations_to_test.append(dm10.Density)
    hascuda = True
except ImportError:
    pass
# We automatically only test the backends available by using the fixtures here


@pytest.fixture(params=implementations_to_test)
def dm(request):
    return request.param(5)


@pytest.fixture(params=implementations_to_test)
def dmclass(request):
    return request.param


@pytest.fixture(params=implementations_to_test)
def dm_random(request):
    n = 5
    a = np.random.random((2**n, 2**n)) * 1j
    a += np.random.random((2**n, 2**n))

    a += a.transpose().conj()
    a = a / np.trace(a)
    dm = request.param(n, a)
    return dm


# Test cases begin here

class TestDensityInit:

    def test_ground_state(self, dmclass):
        dm = dmclass(9)
        assert dm.no_qubits == 9

    def test_can_create_empty(self, dmclass):
        dm = dmclass(0)
        assert dm.no_qubits == 0

    def test_dont_make_huge_matrix(self, dmclass):
        with pytest.raises(ValueError):
            dmclass(200)

    def test_numpy_array(self, dmclass):
        n = 8
        a = np.zeros((2**n, 2**n))
        dm = dmclass(n, a)
        assert dm.no_qubits == n
        assert np.allclose(dm.to_array(), a)


    @pytest.mark.skipif(not hascuda, reason="pycuda not installed")
    def test_gpu_array(self):
        n = 10
        a = ga.zeros(2**(2*n), dtype=np.float64)
        dm = dm10.Density(n, a)
        assert a.gpudata is dm.data.gpudata

    def test_wrong_data(self, dmclass):
        with pytest.raises(ValueError):
            dmclass(10, "bla")

    def test_wrong_size(self, dmclass):
        n = 10
        a = np.zeros((2**n, 2**n))
        with pytest.raises(AssertionError):
            dmclass(n + 1, a)

class TestDensityTrace:

    def test_empty_trace_one(self, dm):
        assert np.allclose(dm.trace(), 1)

    def test_trace_random(self, dmclass):
        n = 5
        a = np.random.random((2**n, 2**n)) * 1j
        a += np.random.random((2**n, 2**n))
        a += a.transpose().conj()

        dm = dmclass(n, a)

        trace_dm = dm.trace()
        trace_np = a.trace()

        assert np.allclose(trace_dm, trace_np)

class TestCopy:

    def test_equality(self, dm):
        dm_copy = dm.copy()
        assert np.allclose(dm.to_array(), dm_copy.to_array())

    @pytest.mark.xfail
    def test_not_equality_after_gate(self, dm):
        dm_copy = dm.copy()
        dm_copy.hadamard(0)
        assert not np.allclose(dm.to_array(), dm_copy.to_array())

class TestDensityGetDiag:

    def test_empty_trace_one(self, dm):
        diag = dm.get_diag()
        diag_should = np.zeros(2**5)
        diag_should[0] = 1
        assert np.allclose(diag, diag_should)

    def test_trace_random(self, dmclass):
        n = 7
        a = np.random.random((2**n, 2**n)) * 1j
        a += np.random.random((2**n, 2**n))

        # make a hermitian
        a += a.transpose().conj()
        dm = dmclass(n, a)

        diag_dm = dm.get_diag()
        diag_a = a.diagonal()

        assert np.allclose(diag_dm, diag_a)

class TestRenormalize:

    def test_renormalize_does_nothing_to_gs(self, dm):
        a0 = dm.to_array()
        dm.renormalize()
        a1 = dm.to_array()
        assert np.allclose(a0, a1)

    def test_random_matrix(self, dmclass):
        n = 6
        a = np.random.random((2**n, 2**n)) * 1j
        a += np.random.random((2**n, 2**n))
        a += a.transpose().conj()
        dm = dmclass(n, a)

        dm.renormalize()
        tr = dm.trace()

        a2 = dm.to_array()

        assert np.allclose(tr, 1)
        assert np.allclose(a2, a / np.trace(a))


@pytest.mark.xfail
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

    def test_squares_to_one(self, dm_random):
        dm = dm_random
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

    # @pytest.mark.skip
    # def test_squares_to_one(self, dm_random):
        # dm = dm_random
        # a0 = dm.to_array()
        # dm.hadamard(4)
        # dm.hadamard(4)
        # # dm.hadamard(2)
        # # dm.hadamard(2)
        # # dm.hadamard(0)
        # # dm.hadamard(0)
        # a1 = dm.to_array()
        # assert np.allclose(np.triu(a0), np.triu(a1))

    def test_squares_to_one_small(self, dmclass):
        dm = dmclass(1)
        print(dm.to_array())

        print(dm._blocksize, dm._gridsize)

        dm.hadamard(0)
        print(dm.to_array())
        dm.hadamard(0)

        print(dm.to_array())

        assert dm.to_array()[0, 0] == 1


@pytest.mark.xfail
class TestDensityRotateX:

    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.rotate_x(10, 1, 0)

    def test_does_something_to_ground_state(self, dm):
        a0 = dm.to_array()
        dm.rotate_x(4, np.cos(0.5), np.sin(0.5))
        a1 = dm.to_array()
        assert not np.allclose(a0, a1)

    def test_excite(self, dmclass):
        dm = dmclass(2)

        dm.rotate_x(0, np.cos(np.pi / 2), np.sin(np.pi / 2))
        dm.rotate_x(1, np.cos(np.pi / 2), np.sin(np.pi / 2))

        a1 = dm.to_array()
        assert np.allclose(np.trace(a1), 1)
        assert np.allclose(a1[-1, -1], 1)

    def test_preserves_trace(self, dm_random):

        assert np.allclose(dm_random.trace(), 1)
        dm_random.rotate_x(2, np.cos(4.2), np.sin(4.2))
        assert np.allclose(dm_random.trace(), 1)

    def test_cubes_to_one(self, dmclass):
        dm = dmclass(1)
        a0 = dm.to_array()
        c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
        dm.rotate_x(0, c, s)
        dm.rotate_x(0, c, s)
        dm.rotate_x(0, c, s)
        a1 = dm.to_array()

        assert np.allclose(a0, a1)


@pytest.mark.xfail
class TestDensityRotateY:

    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.rotate_y(10, 1, 0)

    def test_does_something_to_ground_state(self, dm):
        a0 = dm.to_array()
        dm.rotate_y(4, np.cos(0.5), np.sin(0.5))
        a1 = dm.to_array()
        assert not np.allclose(a0, a1)

    def test_excite(self, dmclass):
        dm = dmclass(2)
        dm.rotate_y(0, np.cos(np.pi / 2), np.sin(np.pi / 2))
        dm.rotate_y(1, np.cos(np.pi / 2), np.sin(np.pi / 2))

        a1 = dm.to_array()
        assert np.allclose(np.trace(a1), 1)
        assert np.allclose(a1[-1, -1], 1)

    def test_cubes_to_one(self, dm):
        a0 = dm.to_array()
        c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
        dm.rotate_y(1, c, s)
        dm.rotate_y(1, c, s)
        dm.rotate_y(1, c, s)
        a1 = dm.to_array()

        assert np.allclose(a0, a1)


@pytest.mark.xfail
class TestDensityRotateZ:

    def test_bit_too_high(self, dm):
        with pytest.raises(AssertionError):
            dm.rotate_z(10, 1, 0)

    def test_does_nothing_to_ground_state(self, dm):
        a0 = dm.to_array()
        dm.rotate_z(4, np.cos(0.5), np.sin(0.5))
        a1 = dm.to_array()
        assert np.allclose(a0, a1)

    def test_excite(self, dmclass):
        dm = dmclass(2)

        dm.hadamard(0)
        dm.rotate_z(0, np.cos(np.pi), np.sin(np.pi))
        dm.hadamard(0)

        dm.hadamard(1)
        dm.rotate_z(1, np.cos(np.pi), np.sin(np.pi))
        dm.hadamard(1)

        a1 = dm.to_array()
        assert np.allclose(np.trace(a1), 1)
        assert np.allclose(a1[-1, -1], 1)

    def test_cubes_to_one(self, dmclass):
        dm = dmclass(1)

        a0 = dm.to_array()

        dm.hadamard(0)
        c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
        dm.rotate_z(0, c, s)
        dm.rotate_z(0, c, s)
        dm.rotate_z(0, c, s)
        dm.hadamard(0)

        a1 = dm.to_array()

        assert np.allclose(a0, a1)


@pytest.mark.xfail
class TestCommutationXYZ:

    def test_excite_deexcite(self, dm):
        a0 = dm.to_array()
        dm.rotate_x(1, np.cos(np.pi / 2), np.sin(np.pi / 2))
        dm.rotate_y(1, np.cos(np.pi / 2), np.sin(np.pi / 2))
        dm.rotate_z(1, np.cos(np.pi), np.sin(np.pi))
        dm.rotate_x(1, np.cos(np.pi / 2), np.sin(np.pi / 2))
        dm.rotate_y(1, np.cos(np.pi / 2), np.sin(np.pi / 2))
        dm.rotate_z(1, np.cos(np.pi), np.sin(np.pi))
        a1 = dm.to_array()
        assert np.allclose(a0, a1)

    def test_pauli_xyz(self, dm):
        a0 = dm.to_array()

        dm.rotate_x(1, np.cos(np.pi / 4), np.sin(np.pi / 4))
        dm.rotate_z(1, np.cos(np.pi / 2), np.sin(np.pi / 2))
        # dm.rotate_x(1, np.cos(4.2), np.sin(4.2))  # should do nothing
        dm.rotate_y(1, np.cos(np.pi / 4), np.sin(np.pi / 4))

        a1 = dm.to_array()
        assert np.allclose(a0, a1)

    def test_xyx(self, dm):

        a0 = dm.to_array()

        dm.rotate_x(1, np.cos(np.pi / 4), np.sin(np.pi / 4))
        dm.rotate_y(1, np.cos(np.pi / 4), np.sin(np.pi / 4))
        dm.rotate_x(1, np.cos(np.pi / 4), -np.sin(np.pi / 4))

        a1 = dm.to_array()

        assert np.allclose(a0, a1)


@pytest.mark.xfail
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

    def test_add_high_ancilla_to_gs_gives_gs(self, dmclass):
        dm = dmclass(9)
        dm2 = dm.add_ancilla(0)
        assert dm2.no_qubits == 10
        assert np.allclose(dm2.trace(), 1)
        a = dm2.to_array()
        assert np.allclose(a[0, 0], 1)

    def test_add_exc_ancilla_to_gs_gives_no_gs(self, dm):
        dm2 = dm.add_ancilla(1)
        assert dm2.no_qubits == dm.no_qubits + 1
        assert np.allclose(dm2.trace(), 1)
        a = dm2.to_array()
        assert np.allclose(a[0, 0], 0)

    def test_preserve_trace_random_state(self, dm_random):
        dm = dm_random
        assert np.allclose(dm.trace(), 1)
        dm2 = dm.add_ancilla(1)
        assert np.allclose(dm2.trace(), 1)


@pytest.mark.xfail
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

    def test_hadamard_gives_50_50_on_small(self, dmclass):
        dm = dmclass(1)

        dm.hadamard(0)
        p0, dm0, p1, dm1 = dm.measure_ancilla(0)

        assert np.allclose(p0, 0.5)
        assert np.allclose(p1, 0.5)

    def test_trace_preserve(self, dm_random):
        dm = dm_random
        p0, dm0, p1, dm1 = dm.measure_ancilla(2)

        assert np.allclose(p0 + p1, 1)

    def test_relax_then_measure_gives_gs(self, dm_random):
        dm = dm_random
        dm.amp_ph_damping(2, 1.0, 1.0)
        p0, dm0, p1, dm1 = dm.measure_ancilla(2)
        assert np.allclose(p1, 0)
        assert np.allclose(p0, 1)

