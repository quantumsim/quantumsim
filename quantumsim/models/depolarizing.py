'''Gateset with a depolarizing channel with a constant error probability.
We put the error channel *after* the gate in each instance.
'''

from quantumsim.circuit import Gate, TwoPTMGate
from quantumsim import ptm
from quantumsim.models import noiseless
import numpy as np


def depolarizing_channel(depol_prob):
    op = np.identity(4)
    for j in range(3):
        op[j, j] = depol_prob
    op = ptm.to_0xy1_basis(op)
    return op


class DepolarizingGate(Gate):
    def __init__(self, depol_prob, **kwargs):
        super().__init__(**kwargs)
        error_ptm = depolarizing_channel(depol_prob)
        if isinstance(self, TwoPTMGate):
            error_ptm = np.kron(error_ptm, error_ptm)
        self.ptm = error_ptm @ ptm


class RotateX(noiseless.RotateX, DepolarizingGate):
    pass
class RotateY(noiseless.RotateY, DepolarizingGate):
    pass
class RotateZ(noiseless.RotateZ, DepolarizingGate):
    pass
class CPhase(noiseless.CPhase, DepolarizingGate):
    pass
class CNOT(noiseless.CNOT, DepolarizingGate):
    pass
class ISwap(noiseless.ISwap, DepolarizingGate):
    pass
class Hadamard(noiseless.Hadamard, DepolarizingGate):
    pass
