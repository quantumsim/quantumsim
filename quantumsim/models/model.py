import inspect
import abc
import numpy as np


class Model(abc.ABCMeta):
    def __init__(self, setup, seed=None):
        self._setup = setup
        self.rng = np.random.RandomState(seed)

    @property
    def setup(self):
        return self._setup

    @property
    def compilers(self):
        return self._compilers

    @staticmethod
    def gate(func, n_qubits=1, plot_metadata=None):
        raise NotImplementedError
        argspec = inspect.getfullargspec(func)
        if argspec.varargs is not None:
            raise RuntimeError(
                "Function passed to Model.gate decorator "
                "must not accept arbitrary arguments"
            )
        if argspec.varkw is not None:
            raise RuntimeError(
                "Function passed to Model.gate decorator "
                "must not accept arbitrary keyword arguments"
            )
        args = argspec.args
        qubit_args = args[:n_qubits]
        if len(qubit_args) != n_qubits:
            raise RuntimeError(
                "Function passed to Model.gate decorator "
                "must accept all its qubits as first arguments"
            )
        free_params = args[n_qubits:]
        return Gate(
            plot_metadata=plot_metadata or {'style': 'box',
                                            'label': func.__name__},
            operation_func=func
        )
