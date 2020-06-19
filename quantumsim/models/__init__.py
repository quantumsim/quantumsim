from .model import Model, WaitingGate
from .library import IdealModel

gates = IdealModel()
# I think this is questionable. I think the gate wrapper
# should be reformated to be indepedent and gates should be a module

__all__ = ["Model", "WaitingGate", "gates"]
