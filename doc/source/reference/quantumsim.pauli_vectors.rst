:mod:`quantumsim.pauli_vectors` -- representation of density matrices
=====================================================================

.. module:: quantumsim.pauli_vectors

Pauli vector interface
----------------------

.. autosummary::
   :toctree: generated/

   pauli_vector.PauliVectorBase

Built-in realizations
---------------------

Quantumsim will export its default PauliVector class implementation as
`quantumsim.PauliVector`. By default CUDA backend is picked, though, for small
number of qubits Numpy backend may be faster.

.. autosummary::
   :toctree: generated/

   PauliVectorNumpy
   PauliVectorCuda

