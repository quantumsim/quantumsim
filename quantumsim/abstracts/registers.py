'''Qubit register classes
'''

import quantumsim as qs
from quantumsim.states import Default as State


class QubitRegister():
    '''QubitRegister - a container for a State class
    that contains absolute references for the qubit indices.

    TODO: add tomography(?)
    '''

    @property
    def qubits(self):
        return self._qubits

    @property
    def n_qubits(self):
        return len(self._qubits)

    @property
    def state(self):
        return self._state

    def __init__(self, qubits, bases=None):
        self._qubits = tuple(qubits)
        if bases is None:
            b = qs.bases.general(2)
            bases = (b.subbasis([0]),) * self.n_qubits
        self._state = State(bases)
