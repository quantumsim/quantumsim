from .model import Model, WaitPlaceholder
from .channel import Channel
from .library import IdealModel

gates = IdealModel()
# I think this is questionable. I think the gate wrapper
# should be reformated to be indepedent and gates should be a module

__all__ = ["Model", "Channel", "WaitPlaceholder", "gates"]
