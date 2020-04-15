from .model import Model, WaitPlaceholder
from .channel import Channel
from .library import IdealModel

gates = IdealModel()

__all__ = ["Model", "Channel", "WaitPlaceholder", "gates"]
