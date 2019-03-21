:mod:`quantumsim.states` -- representation of a state
=====================================================

.. module:: quantumsim.states

State interface
---------------

.. autosummary::
   :toctree: generated/

   state.StateBase

Built-in realizations
---------------------

Quantumsim will export its default State class implementation as
`quantumsim.State`. By default CUDA backend is picked, though, for small
number of qubits Numpy backend may be faster.

.. autosummary::
   :toctree: generated/

   StateNumpy
   StateCuda

