import abc
from quantumsim import Operation


class Channel(metaclass=abc.ABCMeta):
    def __init__(self, qubits=None):
        self._qubits = qubits

    @property
    def channels(self):
        return [self]

    @abc.abstractmethod
    def noise_op(self, qubit, duration, setup=None):
        pass

    @classmethod
    def on(cls, qubits):
        return cls(qubits=qubits)

    @classmethod
    def during(cls, times):
        return cls(times=times)

    def __add__(self, other):
        return MixedChannel(channels=self.channels + other.channels)


class MixedChannel(Channel):
    def __init__(self, channels):
        self._channels = channels

    @property
    def channels(self):
        return self._channels

    def noise_op(self, qubit, duration, setup=None):
        noise_ops = list(
            channel.noise_op(qubit, duration, setup) for channel in self.channels
        )
        return Operation.from_sequence(noise_ops)
