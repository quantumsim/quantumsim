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


no_qubits = 10

x = np.random.random((2**no_qubits, 2**no_qubits)).astype(np.complex128)
x += np.random.random((2**no_qubits, 2**no_qubits))*1j

x = x.T.conj() @ x

x = x/np.trace(x)

def random_dm10():
    "return a random (2**no_qubits, 2**no_qubits) density matrix"
    return x

class TestCphase:
    "test the cphase kernel"
    def test_trace_preserve(self):
        """cphase must preserve trace of density matrix"""
        dm10 = random_dm10()

        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        qbit1 = 9
        qbit2 = 6

        cphase(dm10_gpu, np.uint32((1<<qbit1) | (1<<qbit2)), np.uint32(no_qubits), block=(32,32,1), grid=(32,32,1))

        dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

        assert np.allclose(np.trace(dm10_transformed), 1)
    def test_involution(self):
        """cphase squared must be unity
        """
        dm10 = random_dm10()

        dm10_gpu = drv.to_device(dm10)

        qbit1 = 9
        qbit2 = 6

        cphase(dm10_gpu, np.uint32((1<<qbit1) | (1<<qbit2)), np.uint32(no_qubits), block=(32,32,1), grid=(32,32,1))
        cphase(dm10_gpu, np.uint32((1<<qbit1) | (1<<qbit2)), np.uint32(no_qubits), block=(32,32,1), grid=(32,32,1))

        dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

        assert np.allclose(dm10_transformed, dm10)

class TestHadamard():
    "test the hadamard kernel"
    def test_trace_preserve(self):
        """hadamard must preserve trace"""
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        for qbit in range(no_qubits):

            hadamard(dm10_gpu, np.uint32(1<<qbit), np.float64(0.5), np.uint32(no_qubits),
                    block=(32,32,1), grid=(32,32,1))

            dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

            assert np.allclose(np.trace(dm10_transformed), 1)
    def test_involution(self):
        """hadamard must square to identity"""
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        for qbit in range(no_qubits):

            hadamard(dm10_gpu, np.uint32(1<<qbit), np.float64(0.5), np.uint32(no_qubits),
                    block=(32,32,1), grid=(32,32,1))
            hadamard(dm10_gpu, np.uint32(1<<qbit), np.float64(0.5), np.uint32(no_qubits),
                    block=(32,32,1), grid=(32,32,1))

            dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

            assert np.allclose(dm10_transformed, dm10)

class TestAmpPhDamping:
    "test the amp_ph_damping kernel"
    def test_trace_preserve(self):
        """damping must preserve trace"""
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        assert np.allclose(np.trace(dm10), 1)

        gamma = 0.1
        lamda = 0.0

        gamma = np.float64(gamma)
        s1mgamma = np.float64(np.sqrt(1 - gamma))
        s1mlambda = np.float64(np.sqrt(1 - lamda))


        for qbit in range(no_qubits):
            amp_ph_damping(dm10_gpu, np.uint32(1<<qbit), 
                    gamma, s1mgamma, s1mlambda, np.uint32(no_qubits),
                    block=(32,32,1), grid=(32,32,1))


            dm10_transformed = drv.from_device_like(dm10_gpu, dm10)

            assert np.allclose(np.trace(dm10_transformed), 1)

class TestDMReduce:
    "test the dm_reduce kernel"
    def test_trace_preserve(self):
        "the sum of the traces of the reduces matrices must be one"
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        dm9_0 = np.zeros((2**(no_qubits-1), 2**(no_qubits-1)), np.complex128)
        dm9_1 = np.zeros((2**(no_qubits-1), 2**(no_qubits-1)), np.complex128)

        dm9_0_gpu = drv.to_device(dm9_0)
        dm9_1_gpu = drv.to_device(dm9_1)

        assert np.allclose(np.trace(dm10), 1)

        bit_idx = 2

        dm_reduce(dm10_gpu, np.uint32(bit_idx), dm9_0_gpu, dm9_1_gpu, 
                np.float64(1), np.float64(1), np.uint32(no_qubits),
                block=(32,32,1), grid=(32,32,1))

        dm9_0 = drv.from_device_like(dm9_0_gpu, dm9_0)
        dm9_1 = drv.from_device_like(dm9_1_gpu, dm9_1)

        tr0 = np.trace(dm9_0)
        tr1 = np.trace(dm9_1)

        assert np.allclose(tr0 + tr1, 1)
    def test_full_relax(self):
        "first let the ancilla decay to ground state (using amp_ph_damping), then trace of 0 (1) reduced density matrix must be 1 (0)"
        dm10 = random_dm10()
        dm10_gpu = drv.to_device(dm10)

        dm9_0 = np.zeros((2**(no_qubits-1), 2**(no_qubits-1)), np.complex128)
        dm9_1 = np.zeros((2**(no_qubits-1), 2**(no_qubits-1)), np.complex128)

        dm9_0_gpu = drv.to_device(dm9_0)
        dm9_1_gpu = drv.to_device(dm9_1)

        assert np.allclose(np.trace(dm10), 1)

        bit_idx = 2

        gamma = np.float64(1.0)
        s1mgamma = np.float64(0.0)
        s1mlambda = np.float64(0.0)


        amp_ph_damping(dm10_gpu, np.uint32(1<<bit_idx), 
                gamma, s1mgamma, s1mlambda, np.uint32(no_qubits),
                block=(32,32,1), grid=(32,32,1))
        
        dm_reduce(dm10_gpu, np.uint32(bit_idx), dm9_0_gpu, dm9_1_gpu, 
                np.float64(1), np.float64(1), np.uint32(no_qubits),
                block=(32,32,1), grid=(32,32,1))

        dm9_0 = drv.from_device_like(dm9_0_gpu, dm9_0)
        dm9_1 = drv.from_device_like(dm9_1_gpu, dm9_1)

        tr0 = np.trace(dm9_0)
        tr1 = np.trace(dm9_1)

        assert np.allclose(tr0, 1)
        assert np.allclose(tr1, 0)

class TestGetDiag:
    "test the get_diag kernel"
    def test_get_diag(self):
        "the test_diag kernel must extract the diagonal"

        dm9 = np.random.random((2**(no_qubits-1), 2**(no_qubits-1))).astype(np.complex128)
        dm9_gpu = drv.to_device(dm9)

        diag_dm9 = np.zeros(2**(no_qubits-1), np.float64)
        diag_dm9_gpu = drv.to_device(diag_dm9)

        get_diag(dm9_gpu, diag_dm9_gpu, np.uint32(no_qubits-1), block=(32,1,1), grid=(16,1,1))

        diag_dm9 = drv.from_device_like(diag_dm9_gpu, diag_dm9)

class TestDMInflate:
    "test the dm_inflate kernel"
    def test_inverse_of_reduce(self):
        "first inflating (using dm_inflate) and then reducing must be no-op"

        # make two hermitian dm9
        dm9_0 = np.random.random((2**(no_qubits-1), 2**(no_qubits-1))).astype(np.complex128)
        dm9_0 += 1j*np.random.random((2**(no_qubits-1), 2**(no_qubits-1)))
        dm9_0 = dm9_0 + dm9_0.T.conj() # make hermitian
        dm9_0_gpu = drv.to_device(dm9_0)
        dm9_1 = np.random.random((2**(no_qubits-1), 2**(no_qubits-1))).astype(np.complex128)
        dm9_1 += 1j*np.random.random((2**(no_qubits-1), 2**(no_qubits-1)))
        dm9_1 = dm9_1 + dm9_1.T.conj() # make hermitian
        dm9_1_gpu = drv.to_device(dm9_1)


        # make empty dm10
        dm10 = np.zeros((2**no_qubits, 2**no_qubits), np.complex128)
        dm10_gpu = drv.to_device(dm10)

        bit_idx = 2

        # inflate dm9s to dm10
        dm_inflate(dm10_gpu, np.uint32(bit_idx), dm9_0_gpu, dm9_1_gpu, np.uint32(no_qubits),
                block=(32,32,1), grid=(16,16,1))

        # make two empty dm9s
        dm9_0_p = np.zeros((2**(no_qubits-1), 2**(no_qubits-1)), np.complex128)
        dm9_0_p_gpu = drv.to_device(dm9_0_p)
        dm9_1_p = np.zeros((2**(no_qubits-1), 2**(no_qubits-1)), np.complex128)
        dm9_1_p_gpu = drv.to_device(dm9_1_p)

        # reduce into new dm9s
        dm_reduce(dm10_gpu, np.uint32(bit_idx), dm9_0_p_gpu, dm9_1_p_gpu,
                np.float64(1.0), np.float64(1.0), np.uint32(no_qubits),
                block=(32,32,1), grid=(32,32,1))

        dm9_0_p = drv.from_device_like(dm9_0_p_gpu, dm9_0_p)
        dm9_1_p = drv.from_device_like(dm9_1_p_gpu, dm9_1_p)

        # upper triangle of new dm9s must be original dm9s
        assert np.allclose(np.triu(dm9_0_p), np.triu(dm9_0))
        assert np.allclose(np.triu(dm9_1_p), np.triu(dm9_1))
 
