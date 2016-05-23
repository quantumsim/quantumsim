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
partial_trace = mod.get_function("partial_trace")




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


