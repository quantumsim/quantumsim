import pytest
import numpy as np

import matplotlib.pyplot as plt
hascuda = True
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    import quantumsim.dm10 as dm10

    with open(dm10.kernel_file, "r") as f:
        mod = SourceModule(f.read())

    cphase = mod.get_function("cphase")
    get_diag = mod.get_function("get_diag")

    bit_to_pauli_basis = mod.get_function("bit_to_pauli_basis")
    pauli_reshuffle = mod.get_function("pauli_reshuffle")
    single_qubit_ptm = mod.get_function("single_qubit_ptm")
    trace  = mod.get_function("trace")
except ImportError:
    hascuda = False


no_qubits = 10
block = (32,32,1)
grid = (1<<(no_qubits-5), 1<<(no_qubits-5), 1)

x = np.random.random((2**no_qubits, 2**no_qubits)).astype(np.complex128)
x += np.random.random((2**no_qubits, 2**no_qubits)) * 1j

x = np.dot(x.T.conj(), x)
x = x / np.trace(x)

pytestmark = pytest.mark.skipif(not hascuda, reason="pycuda not installed")

def random_dm10():
    "return a random (2**no_qubits, 2**no_qubits) density matrix"
    return x


class TestToPauli:

    def test_diag_preserve(self):
        dm = random_dm10()

        dm_gpu = drv.to_device(dm)

        for i in range(no_qubits):
            bit_to_pauli_basis(dm_gpu, np.int32(1<<i), np.int32(no_qubits),
                    block=block, grid=grid)

        dm2 = drv.from_device_like(dm_gpu, dm)

        assert np.allclose(np.diag(dm), np.diag(dm2))

    def test_all_real_or_imag(self):
        dm = random_dm10()

        dm_gpu = drv.to_device(dm)

        for i in range(no_qubits):
            bit_to_pauli_basis(dm_gpu, np.int32(1<<i), np.int32(no_qubits),
                    block=block, grid=grid)

        dm2 = drv.from_device_like(dm_gpu, dm)

        where_real = dm2.real == dm2
        where_imag = 1j*dm2.imag == dm2
        where_all = where_real + where_imag

        assert np.all(where_all)

    def test_involution(self):
        dm = random_dm10()

        dm_gpu = drv.to_device(dm)

        for i in range(no_qubits):
            bit_to_pauli_basis(dm_gpu, np.int32(1<<i), np.int32(no_qubits),
                    block=block, grid=grid)

        for i in range(no_qubits):
            bit_to_pauli_basis(dm_gpu, np.int32(1<<i), np.int32(no_qubits),
                    block=block, grid=grid)

        dm2 = drv.from_device_like(dm_gpu, dm)

        assert np.allclose(dm, dm2)

    def test_reshuffle_invertible(self):
        dm = random_dm10()

        dm_gpu = drv.to_device(dm)

        for i in range(no_qubits):
            bit_to_pauli_basis(dm_gpu, np.int32(1<<i), np.int32(no_qubits),
                    block=block, grid=grid)

        dmreal = np.zeros(2**(2*no_qubits))
        dmreal_gpu = drv.to_device(dmreal)

        pauli_reshuffle(dm_gpu, dmreal_gpu, np.int32(no_qubits), np.int32(0),
                block=block, grid=grid)

        dm_gpu2 = drv.mem_alloc(dm.nbytes)
        drv.memset_d8(dm_gpu2, 0, dm.nbytes)

        pauli_reshuffle(dm_gpu2, dmreal_gpu, np.int32(no_qubits), np.int32(1),
                block=block, grid=grid)

        for i in range(no_qubits):
            bit_to_pauli_basis(dm_gpu2, np.int32(1<<i), np.int32(no_qubits),
                    block=block, grid=grid)

        dm2 = drv.from_device_like(dm_gpu2, dm)

        assert np.allclose(dm, dm2)

class TestTrace:
    def test_simple(self):
        n = 32
        x = np.random.random(n);

        x_gpu = drv.to_device(x)
        
        trace(x_gpu, np.int32(-1), block=(n,1,1), grid=(1,1,1), shared = 8*128)

        x2 = drv.from_device_like(x_gpu, x)
        

        assert np.allclose(x2[0], np.sum(x))

    def test_sum_bit0(self):

        n = 32 
        x = np.random.random(n);
        # x = np.arange(n).astype(np.float64)

        x_gpu = drv.to_device(x)
        
        trace(x_gpu, np.int32(0), block=(n,1,1), grid=(1,1,1), shared = 8*128)

        x2 = drv.from_device_like(x_gpu, x)

        print(x)
        print(x2)
        
        assert np.allclose(x2[1], np.sum(x[::2]))
        assert np.allclose(x2[0], np.sum(x[1::2]))

    def test_sum_bit1(self):

        n = 32 
        x = np.random.random(n);
        # x = np.arange(n).astype(np.float64)

        x_gpu = drv.to_device(x)
        
        trace(x_gpu, np.int32(1), block=(n,1,1), grid=(1,1,1), shared = 8*128)

        x2 = drv.from_device_like(x_gpu, x)

        print(x)
        print(x2)
        
        assert np.allclose(x2[1], np.sum(x[::4]) + np.sum(x[1::4]))
        assert np.allclose(x2[0], np.sum(x[2::4]) + np.sum(x[3::4]))

    def test_sum_bit_high(self):

        n = 32 
        x = np.random.random(n);
        # x = np.arange(n).astype(np.float64)

        x_gpu = drv.to_device(x)
        
        trace(x_gpu, np.int32(4), block=(n,1,1), grid=(1,1,1), shared = 8*128)

        x2 = drv.from_device_like(x_gpu, x)

        print(x)
        print(x2)
        
        assert np.allclose(x2[1], np.sum(x[:16]))
        assert np.allclose(x2[0], np.sum(x[16:]))

class TestPTM:

    def test_identity(self):
        ptm = np.eye(4)

        ptm_gpu = drv.to_device(ptm)

        dm = np.random.random((512, 512))


        dm_gpu = drv.to_device(dm)

        single_qubit_ptm(dm_gpu, ptm_gpu, np.int32(2), np.int32(9), block=(512,1,1), grid=(512,1,1), shared = 8*(16 + 512))

        dm2 = drv.from_device_like(dm_gpu, dm)

        assert np.allclose(dm2, dm)

    def test_xy_hadamard_squares_to_one(self):
        ptm = np.array( [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], np.float64)

        ptm_gpu = drv.to_device(ptm)

        dm = np.random.random((512, 512))


        dm_gpu = drv.to_device(dm)

        single_qubit_ptm(dm_gpu, ptm_gpu, np.int32(2), np.int32(9), block=(512,1,1), grid=(512,1,1), shared = 8*(16 + 512))

        dm2 = drv.from_device_like(dm_gpu, dm)
        assert not np.allclose(dm, dm2)

        single_qubit_ptm(dm_gpu, ptm_gpu, np.int32(2), np.int32(9), block=(512,1,1), grid=(512,1,1), shared = 8*(16 + 512))

        dm2 = drv.from_device_like(dm_gpu, dm)

        assert np.allclose(dm2, dm)

    def test_normal_hadamard_squares_to_one(self):

        ptm = np.array( 
                [[0.5, np.sqrt(0.5), 0, 0.5], 
                 [np.sqrt(0.5), 0, 0, -np.sqrt(0.5)], 
                 [0, 0, -1, 0], 
                 [0.5, -np.sqrt(0.5), 0, 0.5]], 
                np.float64)

        ptm_gpu = drv.to_device(ptm)

        dm = np.random.random((512, 512))


        dm_gpu = drv.to_device(dm)

        single_qubit_ptm(dm_gpu, ptm_gpu, np.int32(2), np.int32(9), block=(512,1,1), grid=(512,1,1), shared = 8*(16 + 512))

        dm2 = drv.from_device_like(dm_gpu, dm)
        assert not np.allclose(dm, dm2)

        single_qubit_ptm(dm_gpu, ptm_gpu, np.int32(2), np.int32(9), block=(512,1,1), grid=(512,1,1), shared = 8*(16 + 512))

        dm2 = drv.from_device_like(dm_gpu, dm)

        assert np.allclose(dm2, dm)

