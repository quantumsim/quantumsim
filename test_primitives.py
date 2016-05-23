import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

import matplotlib.pyplot as plt

from pycuda.compiler import SourceModule
with open("./primitives.cu", "r") as f:
        mod = SourceModule(f.read())

cphase = mod.get_function("cphase")
hadamard = mod.get_function("hadamard")
amp_ph_damping = mod.get_function("amp_ph_damping")
dm_reduce = mod.get_function("dm_reduce")
dm_inflate = mod.get_function("dm_inflate")
get_diag = mod.get_function("get_diag")



x = np.random.random((2**10, 2**10)).astype(np.complex128)
x += np.random.random((2**10, 2**10))*1j

x = x.T.conj() @ x

x = x/np.trace(x)

def random_dm10():
    return x


class TestCphase:
    def test_trace_preserve(self):
        """cphase must preserve trace of density matrix"""
        dm10 = random_dm10()

        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        qbit1 = 9
        qbit2 = 6

        cphase(dm10_gpu, np.uint32((1<<qbit1) | (1<<qbit2)), block=(32,32,1), grid=(32,32,1))

        dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

        assert np.allclose(np.trace(dm10_transformed), 1)


    def test_involution(self):
        """cphase squared is unity
        """
        dm10 = random_dm10()

        dm10_gpu = drv.to_device(dm10)

        qbit1 = 9
        qbit2 = 6

        cphase(dm10_gpu, np.uint32((1<<qbit1) | (1<<qbit2)), block=(32,32,1), grid=(32,32,1))
        cphase(dm10_gpu, np.uint32((1<<qbit1) | (1<<qbit2)), block=(32,32,1), grid=(32,32,1))

        dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

        assert np.allclose(dm10_transformed, dm10)

class TestHadamard():
    def test_trace_preserve(self):
        """test if hadamard preserves trace"""
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        for qbit in range(10):

            hadamard(dm10_gpu, np.uint32(1<<qbit), 
                    np.float64(0.5), block=(32,32,1), grid=(32,32,1))

            dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

            assert np.allclose(np.trace(dm10_transformed), 1)


    def test_involution(self):
        """test if hadamard squared is one"""
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        for qbit in range(10):

            hadamard(dm10_gpu, np.uint32(1<<qbit), 
                    np.float64(0.5), block=(32,32,1), grid=(32,32,1))
            hadamard(dm10_gpu, np.uint32(1<<qbit), 
                    np.float64(0.5), block=(32,32,1), grid=(32,32,1))

            dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

            assert np.allclose(dm10_transformed, dm10)

        

class TestAmpPhDamping:
    def test_trace_preserve(self):
        """test if damping preserves trace"""
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        gamma = 0.1
        lamda = 0.0

        gamma = np.float64(gamma)
        s1mgamma = np.float64(np.sqrt(1 - gamma))
        s1mlambda = np.float64(np.sqrt(1 - lamda))


        for qbit in range(10):
            amp_ph_damping(dm10_gpu, np.uint32(1<<qbit), 
                    gamma, s1mgamma, s1mlambda, 
                    block=(32,32,1), grid=(32,32,1))


            dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

            assert np.allclose(np.trace(dm10_transformed), 1)


class TestDMReduce:
    def test_trace_preserve(self):
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        dm9_0 = np.zeros((2**9, 2**9), np.complex128)
        dm9_1 = np.zeros((2**9, 2**9), np.complex128)

        dm9_0_gpu = drv.to_device(dm9_0)
        dm9_1_gpu = drv.to_device(dm9_1)

        assert np.allclose(np.trace(dm10), 1)

        bit_idx = 2

        dm_reduce(dm10_gpu, np.uint32(bit_idx), dm9_0_gpu, dm9_1_gpu, 
                np.float64(1), np.float64(1),
                block=(32,32,1), grid=(32,32,1))

        dm9_0 = drv.from_device_like(dm9_0_gpu, dm9_0)
        dm9_1 = drv.from_device_like(dm9_1_gpu, dm9_1)

        tr0 = np.trace(dm9_0)
        tr1 = np.trace(dm9_1)

        assert np.allclose(tr0 + tr1, 1)

    def test_full_relax(self):
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        dm9_0 = np.zeros((2**9, 2**9), np.complex128)
        dm9_1 = np.zeros((2**9, 2**9), np.complex128)

        dm9_0_gpu = drv.to_device(dm9_0)
        dm9_1_gpu = drv.to_device(dm9_1)

        assert np.allclose(np.trace(dm10), 1)

        bit_idx = 2

        gamma = np.float64(1.0)
        s1mgamma = np.float64(0.0)
        s1mlambda = np.float64(0.0)


        amp_ph_damping(dm10_gpu, np.uint32(1<<bit_idx), 
                gamma, s1mgamma, s1mlambda, 
                block=(32,32,1), grid=(32,32,1))
        
        dm_reduce(dm10_gpu, np.uint32(bit_idx), dm9_0_gpu, dm9_1_gpu, 
                np.float64(1), np.float64(1),
                block=(32,32,1), grid=(32,32,1))

        dm9_0 = drv.from_device_like(dm9_0_gpu, dm9_0)
        dm9_1 = drv.from_device_like(dm9_1_gpu, dm9_1)

        tr0 = np.trace(dm9_0)
        tr1 = np.trace(dm9_1)

        assert np.allclose(tr0, 1)


class TestGetDiag:
    def test_get_diag(self):

        dm9 = np.random.random((2**9, 2**9)).astype(np.complex128)
        dm9_gpu = drv.to_device(dm9)

        diag_dm9 = np.zeros(2**9, np.float64)
        diag_dm9_gpu = drv.to_device(diag_dm9)

        get_diag(dm9_gpu, diag_dm9_gpu, block=(32,1,1), grid=(16,1,1))

        diag_dm9 = drv.from_device_like(diag_dm9_gpu, diag_dm9)

class TestDMInflate:
    def test_inverse_of_reduce(self):
        dm9_0 = np.random.random((2**9, 2**9)).astype(np.complex128)
        dm9_0 += 1j*np.random.random((2**9, 2**9))
        dm9_0 = dm9_0 + dm9_0.T.conj() # make hermitian
        dm9_0_gpu = drv.to_device(dm9_0)
        dm9_1 = np.random.random((2**9, 2**9)).astype(np.complex128)
        dm9_1 += 1j*np.random.random((2**9, 2**9))
        dm9_1 = dm9_1 + dm9_1.T.conj() # make hermitian
        dm9_1_gpu = drv.to_device(dm9_1)


        dm10 = np.zeros((2**10, 2**10), np.complex128)
        dm10_gpu = drv.to_device(dm10)

        bit_idx = 2

        dm_inflate(dm10_gpu, np.uint32(bit_idx), dm9_0_gpu, dm9_1_gpu, 
                block=(32,32,1), grid=(16,16,1))

        dm9_0_p = np.zeros((2**9, 2**9), np.complex128)
        dm9_0_p_gpu = drv.to_device(dm9_0_p)
        dm9_1_p = np.zeros((2**9, 2**9), np.complex128)
        dm9_1_p_gpu = drv.to_device(dm9_1_p)

        dm_reduce(dm10_gpu, np.uint32(bit_idx), dm9_0_p_gpu, dm9_1_p_gpu,
                np.float64(1.0), np.float64(1.0),
                block=(32,32,1), grid=(32,32,1))

        dm9_0_p = drv.from_device_like(dm9_0_p_gpu, dm9_0_p)
        dm9_1_p = drv.from_device_like(dm9_1_p_gpu, dm9_1_p)

        assert np.allclose(np.triu(dm9_0_p), np.triu(dm9_0))
        assert np.allclose(np.triu(dm9_1_p), np.triu(dm9_1))
 
