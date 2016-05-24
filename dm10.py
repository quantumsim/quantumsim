import numpy as np
import pycuda.driver as drv

class Density10:
    def __init__(self, d9, a):
        """create a dm10 by adding ancilla in state a to a dm9
        dm9: a Density9 describing the state of 9 data qubits
        a: 1 or 0, the state of the ancilla added
        """
        pass
    
    def cphase(self, bit1, bit2):
        """Apply a cphase gate between bit1 and bit2
        bit1, bit2: integer between 0 and 9. 9 is the ancilla. "a" is a synonym for 9.
        """
        pass

    def hadamard(self, bit):
        """Apply a hadamard gate to bit.
        bit: integer between 0 and 9, or "a".
        """
        pass

    def amp_ph_damping(self, bit, params):
        """Apply a amplitude and phase damping channel to bit.
        bit: integer between 0 and 9, or "a".
        params: the damping probabilities (gamma, lambda)
        """
        pass


    def meas(self):
        """Measure the qubit. Return two unnormalized Density9 matrices with 
        traces corresponding to probabilities.
        """
        return (d9_0, d9_1)


class Density9:
    def __init__(self, data=None):
        """A density matrix describing the state of 9 data qubits
        data: a gpu array containing the dense density matrix. If None, 
              creata an initial density matrix with all qubits in ground state.
        """
        drv.


    def trace(self):
        """Trace of the density matrix."""
        return tr
