from ..bases import PauliBasis

class State:
    """Class, that represents a state of a quantum system.

    Parameters
    ----------
    dim : int or list of int
        Dimentionality of a system. If a single integer is given,
        is interpreted as number of qubits in a state. If a list of
        integers is given, is interpreted as a Hilbert dimentionality list
        for all qubits involved.
    """

    def __init__(self, dim):
        if hasattr(dim, '__iter__'):
            self._bases = [PauliBasis(n) for n in dim]
        else:
            self._bases = [PauliBasis(2)] * dim

    def probability(self, *indices, axis='z'):
        """Returns a probability of the measurement, not actually performing it.

        Parameters
        ----------
        i0, i1, ..., iN: int
            Indices of qubits of interest in the state. If none of them are
            specified, all of qubits in the state are returned.
        axis: str
            Basis, in which a measurement should be performed. Default is `z`.

        Returns
        -------
        list of tuples
            Probabilities for qubits to stay in a certain state (0, 1, 2, ...),
            dependent on the dimensionality of qubits' basis.
        """
        raise NotImplementedError()
