:mod:`quantumsim.circuit` -- cirquit construction routines
==========================================================

.. module:: quantumsim

Cirquit
-------

.. autosummary::
   :toctree: generated/

   quantumsim.circuit.Circuit

Qubits
------

.. autosummary::
   :toctree: generated/

   quantumsim.circuit.ClassicalBit
   quantumsim.circuit.VariableDecoherenceQubit

Gates
-----

.. autosummary::
   :toctree: generated/

   quantumsim.circuit.RotateX
   quantumsim.circuit.RotateY
   quantumsim.circuit.RotateZ
   quantumsim.circuit.RotateEuler
   quantumsim.circuit.Hadamard
   quantumsim.circuit.AmpPhDamp
   quantumsim.circuit.DepolarizingNoise
   quantumsim.circuit.BitflipNoise
   quantumsim.circuit.ButterflyGate
   quantumsim.circuit.CPhase
   quantumsim.circuit.CNOT
   quantumsim.circuit.ISwap
   quantumsim.circuit.ISwapRotation
   quantumsim.circuit.Swap
   quantumsim.circuit.CPhaseRotation
   quantumsim.circuit.Measurement
   quantumsim.circuit.ResetGate
   quantumsim.circuit.ConditionalGate
   quantumsim.circuit.ClassicalCNOT
   quantumsim.circuit.ClassicalNOT

Abstract classes for gates and qubits
-------------------------------------

.. autosummary::
   :toctree: generated/

   quantumsim.circuit.Qubit
   quantumsim.circuit.Gate
   quantumsim.circuit.SinglePTMGate
   quantumsim.circuit.TwoPTMGate

Samplers
--------

.. autosummary::
   :toctree: generated/

   quantumsim.circuit.selection_sampler
   quantumsim.circuit.uniform_sampler
   quantumsim.circuit.uniform_noisy_sampler
   quantumsim.circuit.BiasedSampler
