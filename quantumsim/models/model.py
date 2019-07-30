import inspect
import abc
import numpy as np

from quantumsim import Operation
from quantumsim.circuits import TimeAgnosticGate, TimeAwareGate, \
    FinalizedCircuit
from ..operations import Placeholder
from .. import bases


from ..operations.operation import IndexedOperation


class WaitPlaceholder(Placeholder):
    def __init__(self, time, dim):
        super().__init__((bases.general(dim),))
        self.time = time


class Model(metaclass=abc.ABCMeta):
    """
    Parameters
    ----------
    setup : quantumsim.Setup
    seed : int
        Seed for initializing an internal random number generator.
    """
    def __init__(self, setup, seed=None):
        self._setup = setup
        self.rng = np.random.RandomState(seed)

    def wait(self, qubit, time):
        return WaitPlaceholder(time, self.dim).at(qubit)

    def p(self, param, *qubits):
        return self._setup.param(param, qubits)

    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @staticmethod
    def _normalize_operation(op, qubits):
        out = []
        if isinstance(op, Operation):
            if len(qubits) > 1:
                raise ValueError(
                    "Can't construct an operation from a sequence: "
                    "all operations in multi-qubit gate must provide "
                    "indices using Operation.at() method.")
            return op.at(0)
        elif isinstance(op, IndexedOperation):
            op, ix = op
            ix = tuple(qubits.index(qubit) for qubit in ix)
            return op.at(*ix)
        else:
            raise ValueError(
                "Sequence of operations contains an object of type {}, "
                "while it may contain only Operation or IndexedOperation"
                .format(type(op)))

    @staticmethod
    def gate(duration=None, plot_metadata=None):
        def gate_decorator(func):
            def make_operation(self, *qubits, **params):
                sequence = func(self, *qubits)
                sequence = sequence if hasattr(sequence, '__iter__') else \
                    (sequence,)
                sequence = [self._normalize_operation(op, qubits) for op
                            in sequence]
                return Operation.from_sequence(sequence, qubits) \
                    .substitute(**params)

            if duration is None:
                def wrapper(self, *qubits, **params):
                    return TimeAgnosticGate(
                        qubits, make_operation(self, *qubits, **params),
                        plot_metadata)
            else:
                def wrapper(self, *qubits, **params):
                    if callable(duration):
                        _duration = duration(*qubits, self._setup)
                    elif isinstance(duration, str):
                        _duration = self.p(duration, *qubits)
                    else:
                        _duration = duration
                    return TimeAwareGate(
                        qubits, make_operation(self, *qubits, **params),
                        _duration, 0., plot_metadata)
            wrapper.__name__ = func.__name__
            return wrapper
        return gate_decorator

    def finalize(self, circuit, bases_in=None):
        """
        This function is aimed to perform post-processing operations on a
        circuit (such as, for example, adding waiting gates) and returns a
        finalized version of this circuit.

        Parameters
        ----------
        circuit: quantumsim.circuits.CircuitBase
            A circuit
        bases_in: tuple of quantumsim.PauliBasis

        Returns
        -------
        quantumsim.circuits.FinalizedCircuit
            A post-processed and finalized version of the circuit.
        """
        return FinalizedCircuit(circuit, bases_in=bases_in)
