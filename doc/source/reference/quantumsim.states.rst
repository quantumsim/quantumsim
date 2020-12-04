:mod:`quantumsim.states` -- representation of density matrices
=====================================================================

.. module:: quantumsim.states

Pauli vector interface
----------------------

.. autosummary::
   :toctree: generated/

   state.State

Built-in realizations
---------------------

Quantumsim will export its default PauliVector class implementation as
`quantumsim.State`. By default CUDA backend is picked, though, for small
number of qubits Numpy backend may be faster.

.. autosummary::
   :toctree: generated/

   StateNumpy
   StateCuda
