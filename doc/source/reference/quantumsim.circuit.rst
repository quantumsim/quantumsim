:mod:`quantumsim.circuit` -- cirquit construction routines
==========================================================

.. module:: quantumsim.circuit

Abstract classes for gates and qubits
-------------------------------------

.. autosummary::
   :toctree: generated/

   Qubit
   Gate
   SinglePTMGate
   TwoPTMGate

Cirquit
-------

.. autosummary::
   :toctree: generated/

   Circuit

Qubits
------

.. autosummary::
   :toctree: generated/

   ClassicalBit
   VariableDecoherenceQubit

Gates
-----

.. autosummary::
   :toctree: generated/

   RotateX
   RotateY
   RotateZ
   RotateEuler
   Hadamard
   AmpPhDamp
   DepolarizingNoise
   BitflipNoise
   ButterflyGate
   CPhase
   CNOT
   ISwap
   ISwapRotation
   Swap
   CPhaseRotation
   Measurement
   ResetGate
   ConditionalGate
   ClassicalCNOT
   ClassicalNOT

Samplers
--------

.. autosummary::
   :toctree: generated/

   selection_sampler
   uniform_sampler
   uniform_noisy_sampler
   BiasedSampler
