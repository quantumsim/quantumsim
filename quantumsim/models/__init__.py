from .model import Model, WaitingGate
from .setup import Setup
from .library import PerfectQubitModel, PerfectQutritModel

perfect_qubits = PerfectQubitModel()
perfect_qutrits = PerfectQutritModel()

__all__ = ["Model", "Setup", "WaitingGate", "perfect_qubits", "perfect_qutrits"]
