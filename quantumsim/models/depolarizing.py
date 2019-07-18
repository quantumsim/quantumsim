'''Gateset with a depolarizing channel with a constant error probability.
We put the error channel *after* the gate in each instance.
'''

from quantumsim.circuit import Gate, SinglePTMGate, TwoPTMGate
from quantumsim import ptm
from quantumsim.models import noiseless
import numpy as np


def depolarizing_channel(depol_noise):
    op = np.identity(4)
    for j in range(3):
        op[j+1, j+1] *= (1-depol_noise)
    op = ptm.to_0xy1_basis(op)
    return op


class DepolarizingSinglePTMGate(SinglePTMGate):
    def __init__(self, bit, time, depol_noise, **kwargs):
        super().__init__(bit, time, **kwargs)
        error_ptm = depolarizing_channel(depol_noise)
        self.ptm = error_ptm @ self.ptm

class DepolarizingTwoPTMGate(TwoPTMGate):
    def __init__(self, bit0, bit1, time, depol_noise, **kwargs):
        super().__init__(bit0, bit1, time, **kwargs)
        error_ptm = depolarizing_channel(depol_noise)
        error_ptm = np.kron(error_ptm, error_ptm)
        self.two_ptm = error_ptm @ self.two_ptm

class RotateX(DepolarizingSinglePTMGate, noiseless.RotateX):
    pass
class RotateY(DepolarizingSinglePTMGate, noiseless.RotateY):
    pass
class RotateZ(DepolarizingSinglePTMGate, noiseless.RotateZ):
    pass
class CPhase(DepolarizingTwoPTMGate, noiseless.CPhase):
    pass
class CNOT(DepolarizingTwoPTMGate, noiseless.CNOT):
    pass
class ISwap(DepolarizingTwoPTMGate, noiseless.ISwap):
    pass
class Hadamard(DepolarizingSinglePTMGate, noiseless.Hadamard):
    pass
