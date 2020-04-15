import abc
from quantumsim import Operation


class Channel:
    @property
    @abc.abstractmethod
    def channels(self):
        """
        A list of the noise channels present in the general channel
        """
        pass

    @abc.abstractmethod
    def noise_op(self, qubit, duration):
        """
        Return an evaluated quantumsim.Operation corresponding to the noise process

        Parameters
        ----------
        qubit : str
            Thq qubit label for which the noise is applied
        duration : float
            The duration of the noise process
        """
        pass

    def __add__(self, other):
        """
        Combines two noise channels that are present in the experiment

        Parameters
        ----------
        other : quantumsim.models.Channel
            The other channel that is active

        Returns
        -------
        quantumsim.models.Channel
            The mixture of the two channels
        """
        return MixedChannel(channels=self.channels + other.channels)

    @staticmethod
    def from_operator(noise_op, qubits=None):
        """
        Initialized a noise channel from given noise_operator function

        Parameters
        ----------
        noise_op : function
            The function that returns that takes as  arguements the qubits and duration
            and return the corresponding noise operator.
        qubits : tuple, optional
            The qubits to which the noise channel is constrained, by default None

        Returns
        -------
        quantumsim.model.NoiseChannel
            The noise channel acting on the qubits
        """
        return NoiseChannel(noise_op, qubits)


class NoiseChannel(Channel):
    def __init__(self, noise_op, qubits=None):
        """
        A general noise channel.

        The noise channel returns a noise operator, corresponding to the noise process.
        This operator currently is expected to be a function of the qubit or duration of the process.
        TODO: Expand this to fixed operators (such as what one might expected for depolarizing type noise)
        The channel can be be specific to some qubits and/or only be applied during specific periods of time.

        Parameters
        ----------
        noise_op : function
            The function, that takes the qubit and duration as arguements
            and return a single-qubit operation corresponding to the noise process
        qubits : tuple, optional
            The qubits on which the channel acts on, by default None
        """
        self._noise_op = noise_op
        self._qubits = qubits

    @property
    def channels(self):
        return [self]

    def noise_op(self, qubit, duration):
        op = self._noise_op(qubit, duration)
        return op

    def on(self, qubits):
        return NoiseChannel(self._noise_op, qubits=qubits)


class MixedChannel(Channel):
    def __init__(self, channels):
        """
        A Mixed channel, containing two or more noise channels.

        Parameters
        ----------
        channels : list of quantumsim.models.Channel instances
            The list of simultaneously active noise channels
        """
        self._channels = channels

    @property
    def channels(self):
        return self._channels

    def noise_op(self, qubit, duration):
        noise_ops = list(channel.noise_op(qubit, duration) for channel in self.channels)
        return Operation.from_sequence(noise_ops)
