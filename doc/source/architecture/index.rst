Mathematical concepts and conventions
=====================================

Internal data representation of Quantumsim uses superoperator formalism
[1]_ [2]_. We store states as vectors in Hilbert-Schmidt space of dimension
:math:`d^2`, where :math:`d = \sum_{i=1}^{N} d_i` is a sum of dimensionalities
of all qudits in a system. This is done by expanding density matrix and quantum
operations in a basis of Hermitian matrices (see
:ref:`Pauli Bases <Pauli Bases>`).
In this representation, quantum state :math:`\rho` is a real-valued vector,
and quantum operation is a real-valued matrix.

.. toctree::
   :maxdepth: 2

   pauli
   ptm_manipulation
   lindblad


.. [1] A. Y. Kitaev, A. H. Shen, and M. N. Vyalyi, "Classical and Quantum
       Computation" (American Mathematical Society, 2002).

.. [2] D. Greenbaum, "Introduction to Quantum Gate Set tomography",
       `arXiv:1509.02921 <https://arxiv.org/abs/1509.02921>`_.

