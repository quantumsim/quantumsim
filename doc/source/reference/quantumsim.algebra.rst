:mod:`quantumsim.algebra` -- common algebraic manipulations
===========================================================

.. module:: quantumsim.algebra

Basic algebraic manipulations
-----------------------------

.. autosummary::
   :toctree: generated/

   kraus_to_ptm
   ptm_convert_basis
   dm_to_pv
   pv_to_dm

Miscelaneous
------------
.. autosummary::
   :toctree: generated/

   tools.random_hermitian_matrix
   tools.random_unitary_matrix
   tools.verify_kraus_unitarity
   tools.verify_ptm_trace_pres

Pauli matrices
--------------

.. py:attribute:: sigma
   :annotation: dict

   Pauli matrices dictionary (`'I'` is identity, `'X'` is :math:`\sigma_x`,
   `'Y'` is :math:`\sigma_y`, `'Z'` is :math:`\sigma_z`).