:mod:`quantumsim.ptm` -- Pauli transfer matrix routines
=======================================================

.. module:: quantumsim.ptm

Base classes
---------------------

.. autosummary::
   :toctree: generated/

   PauliBasis
   PTM
   TwoPTM

Pauli transfer matrix compiler
------------------------------

.. autosummary::
   :toctree: generated/

   CompilerBlock
   TwoPTMCompiler

Library of Pauli bases
----------------------

.. autosummary::
   :toctree: generated/

   GeneralBasis
   PauliBasis_0xy1
   PauliBasis_ixyz
   GellMannBasis

Library of single-qubit Pauli transfer matrices
-----------------------------------------------

.. autosummary::
   :toctree: generated/

   ExplicitBasisPTM
   LinearCombPTM
   ProductPTM
   ConjunctionPTM
   PLMIntegrator
   AdjunctionPLM
   LindbladPLM
   RotateXPTM
   RotateYPTM
   RotateZPTM
   AmplitudePhaseDampingPTM

Library of two-qubit Pauli transfer matrices
--------------------------------------------

.. autosummary::
   :toctree: generated/

   TwoPTMProduct
   TwoKrausPTM
   CPhaseRotationPTM
   TwoPTMExplicit

Compatibility layer
-------------------

.. autosummary::
   :toctree: generated/

   general_ptm_basis_vector
   to_0xy1_basis
   to_0xyz_basis
   hadamard_ptm
   amp_ph_damping_ptm
   gen_amp_damping_ptm
   dephasing_ptm
   bitflip_ptm
   rotate_x_ptm
   rotate_y_ptm
   rotate_z_ptm
   single_kraus_to_ptm
   single_kraus_to_ptm_general
   double_kraus_to_ptm
