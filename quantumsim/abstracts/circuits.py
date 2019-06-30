"""Circuit classes
"""

from quantumsim import Operation
import quantumsim as qs


class Circuit():

    '''The base circuit object that the user sees.

    A circuit is an object that is constructed by a builder,
    with a fixed order of the qubits within, and may be
    applied to a QubitRegister
    '''

    @property
    def qubits(self):
        return self._qubits

    @property
    def n_qubits(self):
        return len(self._qubits)
    

    def __call__(self, qr):
        '''
        Applies the operation to the state within a qubit register.

        Parameters
        -----
        qr : QubitRegister

        Raises
        -----
        AttributeError : if self.operation doesn't exist.
        '''

        indices = [qr.qubits.index(q) for q in self.qubits]
        try:
            self.operation(qr.state, *indices)
        except AttributeError:
            raise AttributeError(
                'I dont have an operation yet. This probably means'
                'that I have not been properly compiled.')

    def __init__(self, qubits, *, operation=None):

        self._qubits = qubits
        if operation:
            self.operation = operation


class SimpleCircuit(Circuit):
    '''A circuit that takes in individual and grouped
    operation objects, stores them in a list, and then
    compiles the result upon request
    '''

    def __init__(self, qubits, gates=None):
        super().__init__(qubits)
        self.gates = gates or []

    def add_gate(self, gate):
        '''Adds a single operation to the list of gates

        Parameters
        -----
        gate : Operation
        '''

        self.gates.append(gate)


    def add_sequence(self, gates):
        '''Adds multiple operations to the list of gates

        Parameters
        -----
        gates : list of Operations
        '''

        self.gates += gates


    def compile(self, bases=None):
        '''Compiles current list of gates to make operation.

        Parameters
        -----
        basis : the input bases. If none, takes standard
            qubit basis
        '''
        if bases is None:
            bases = (qs.bases.general(2).subbasis([0]),) * self.n_qubits
        self.operation = Operation.from_sequence(
            self.gates).compile(bases_in=bases)

