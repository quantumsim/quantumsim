from .model import Model, WaitPlaceholder
from .channel import Channel
from .library import IdealModel, AmpDampChannel

gates = IdealModel()
amp_damp_channel = AmpDampChannel()

__all__ = ["Model", "Channel", "WaitPlaceholder", "gates", "amp_damp_channel"]
