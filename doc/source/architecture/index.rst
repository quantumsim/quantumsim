Quantumsim Architecture Overview
================================

Internal data representation of Quantumsim uses superoperator formalism
[1]_ [2]_. We store states as vectors in Hilbert-Schmidt space of dimension
:math:`d^2`, where :math:`d = \sum_{i=1}^{N} d_i` is a sum of dimensionalities
of all qudits in a system. This is done by expanding density matrix and quantum
operations in `Pauli basis`_. In this representation, quantum state :math:`\rho`
is a real-valued vector, and quantum operation is a real-valued matrix.

Pauli Basis
-----------

The simpilest example of a Pauli basis is a basis, that consists of four
matrices :math:`\left\{ I,\ \sigma_x,\ \sigma_y,\ \sigma_z \right\}` of unit
matrices and three Pauli matrices.
This is a full basis in a vector space of :math:`2 \times 2` complex Hermitian
matrices.
Therefore, a state of a two-level system with a density matrix
:math:`\hat{\rho}` can be represented as a vector
:math:`\left| \rho \right\rangle` with components :math:`\rho_i` as follows:

.. math::

    \hat{\rho} = \sum_{i=0}^3 \rho_i \hat{P}_i,

where :math:`\hat{P}_i` are the elements of the basis above. Coefficients of
this vector can be determined as:

.. math::

    \rho_i = \frac12 \text{Tr} \hat{P}_i \hat{\rho}


.. [1] A. Y. Kitaev, A. H. Shen, and M. N. Vyalyi, "Classical and Quantum
       Computation" (American Mathematical Society, 2002).

.. [2] D. Greenbaum, "Introduction to Quantum Gate Set tomography",
       `arXiv:1509.02921 <https://arxiv.org/abs/1509.02921>`_.

