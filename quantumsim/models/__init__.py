from .model import Model, WaitingGate
from .library import PerfectQubitModel, PerfectQutritModel

perfect_qubits = PerfectQubitModel()
perfect_qutrits = PerfectQutritModel()

__all__ = ["Model", "WaitingGate", "perfect_qubits", "perfect_qutrits"]
